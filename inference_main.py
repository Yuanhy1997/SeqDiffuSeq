"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import os, json
from typing import List
import numpy as np
import torch as th
import torch.distributed as dist
from transformers import set_seed
from src.utils import dist_util, logger

from args_utils import *
from model_utils import create_model_and_diffusion
from args_utils import create_argparser, args_to_dict, model_and_diffusion_defaults
from tokenizer_utils import create_tokenizer
import dataloader_utils
from mpi4py import MPI

def main():

    args = create_argparser().parse_args()

    set_seed(args.seed)
    th.manual_seed(args.seed)
    print(args.seed)
    dist_util.setup_dist()
    logger.configure()

    # load configurations.
    args.checkpoint_path = os.path.split(args.model_name_or_path)[0]

    config_path = os.path.join(args.checkpoint_path, "training_args.json")
    training_args = read_training_args(config_path)
    training_args["batch_size"] = args.batch_size
    training_args["diffusion_steps"] = args.diffusion_steps
    training_args['model_name_or_path'] = args.model_name_or_path
    training_args["clamp"] = args.clamp
    training_args['out_dir'] = args.out_dir
    training_args['num_samples'] = args.num_samples
    training_args['val_txt_path'] = args.val_txt_path
    training_args['top_p'] = args.top_p
    training_args['sequence_len_src'] = args.sequence_len_src
    training_args['sequence_len'] = args.sequence_len
    training_args['generate_by_q'] = args.generate_by_q
    training_args['generate_by_mix'] = args.generate_by_mix
    training_args['time_schedule_path'] = args.time_schedule_path
    training_args['seed'] = args.seed
    
    args.__dict__.update(training_args)
    args.sigma_small = True

        
    logger.info(f"Init pretrained = {args.init_pretrained}")
    logger.info(f"Freeze embeddings = {args.freeze_embeddings}")
    logger.info(f"Use pretrained embeddings = {args.use_pretrained_embeddings}")
    logger.info(f"Use pretrained embeddings = {args.use_pretrained_tokenizer}")
    
    tokenizer = create_tokenizer(return_pretokenized=args.use_pretrained_tokenizer,
                                 path=f"data/{args.dataset}/",
                                 tokenizer_type='byte-level',
                                 tokenizer_ckpt=args.pretrained_tokenizer)
    
    model, diffusion = create_model_and_diffusion(
        pad_tok_id=tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') else tokenizer.get_vocab()['<pad>'],
        resume_checkpoint=args.resume_checkpoint, **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    diffusion._load_time_schedule(args.time_schedule_path)
    model.load_state_dict(dist_util.load_state_dict(args.model_name_or_path, map_location="cpu"))
    model.eval()

    print('data path', args.val_txt_path)
    val_dataloader = dataloader_utils.get_dataloader(
        tokenizer=tokenizer,
        args=args,
        data_path=args.val_txt_path,
        batch_size=args.batch_size,
        max_seq_len=args.sequence_len,
        max_seq_len_src=args.sequence_len_src,
    )

    if args.num_samples <= 0:
        args.num_samples = len(dataloader_utils.TextDataset_translation(tokenizer=tokenizer, data_path=args.val_txt_path, source=args.src, target=args.tgt,
                                                                        shard=MPI.COMM_WORLD.Get_rank(),
                                                                        num_shards=MPI.COMM_WORLD.Get_size()))
        logger.log(f"sample count is {args.num_samples}")
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.log(f"the parameter count is {pytorch_total_params}")

    diffusion.rescale_timesteps = True

    model.to(dist_util.dev())
    model.eval()  # DEBUG

    logger.log("sampling...")
    logger.log(f"Clamping is set to {args.clamp}")
    all_samples = []
    ground_true_samples = []
    while len(all_samples) * args.batch_size < args.num_samples:
        batch, _ = next(val_dataloader)
        model_kwargs = {key:item.to(dist_util.dev()) for key, item in batch.items() if 'decoder' not in key}
        sample_shape = (args.batch_size, args.sequence_len, model.input_transformers.shared.weight.shape[1])
        print('sample_shape', sample_shape)
        sample = diffusion.p_sample_loop(
                model,
                sample_shape,
                clip_denoised=args.clip_denoised,
                denoised_fn=None,
                model_kwargs=model_kwargs,
                top_p=args.top_p,
                progress=True,
                tokenizer=tokenizer,
                log_verbose=True,
                decoder_inputs=batch['decoder_input_ids'],
                generate_by_q=args.generate_by_q,
                generate_by_mix=args.generate_by_mix,
                generate_by_mix_prob=args.generate_by_mix_prob,
                generate_by_mix_part=args.generate_by_mix_part,
            )

        logits = model.get_logits(sample)  # bsz, seqlen, vocab
        cands = th.topk(logits, k=1, dim=-1).indices.squeeze()
        if args.decoder_attention_mask:
            cands[model_kwargs['decoder_attention_mask']==0] = 1

        gathered_samples = [th.zeros_like(cands) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, cands)  # gather not supported with NCCL
        all_samples.extend([sample.cpu().numpy() for sample in gathered_samples])
        print('number of sample', len(all_samples), all_samples[0].shape)

        batch['decoder_input_ids'] = batch['decoder_input_ids'].to(dist_util.dev())
        gathered_ground_true_sample = [th.zeros_like(batch['decoder_input_ids']) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_ground_true_sample, batch['decoder_input_ids'])
        ground_true_samples.extend([sample.cpu().numpy() for sample in gathered_ground_true_sample])

        logger.log(f"created {len(all_samples) * args.batch_size} samples")

    cands = np.concatenate(all_samples, axis=0)
    cands = cands[: args.num_samples]

    decoded_sentences = []
    for seq in cands:
        seq = seq[seq>2]
        decoded_sentence = tokenizer.decode(seq.tolist(), skip_special_tokens=True)
        decoded_sentences.append(decoded_sentence)
    
    ground_true_sentences = []
    ground_true_samples = np.concatenate(ground_true_samples, axis=0)[: args.num_samples]
    for seq in ground_true_samples:
        seq = seq[seq>2]
        ground_true_sentence = tokenizer.decode(seq.squeeze().tolist(), skip_special_tokens=True)
        ground_true_sentences.append(ground_true_sentence)

    dist.barrier()
    logger.log("sampling complete")

    write_outputs(args=args, 
                  sentences=decoded_sentences, 
                  gt_sentences = ground_true_sentences,
                  raw_sentences=cands,
                  raw_gt_sentences=ground_true_samples,)


def load_embeddings(checkpoint_path, tokenizer, emb_dim):
    embeddings = th.nn.Embedding(tokenizer.vocab_size, emb_dim)
    embeddings.load_state_dict(th.load(f'{checkpoint_path}/random_emb.torch'))
    return embeddings


def read_training_args(config_path):
    with open(config_path, "r") as f:
        return json.load(f)


def write_outputs(args: dict, sentences: List[str], gt_sentences: List[str], raw_sentences, raw_gt_sentences) -> None:

    model_dir = os.path.split(args.model_name_or_path)[0]
    model_base_name = os.path.split(args.model_name_or_path)[1]
    if args.generate_by_q:
        comments = f'predict_by_qsample_{args.seed}'
    elif args.generate_by_mix:
        comments = f'predict_by_mixsample_{args.generate_by_mix_prob}_{args.generate_by_mix_part}_{args.seed}'
    else:
        comments = f'normal_{args.seed}'
    num_samples = len(sentences)
    output_file_basepath = os.path.join(
        model_dir,
        f"{model_base_name}.samples_{num_samples}.steps-{args.diffusion_steps}.clamp-{args.clamp}-{comments}",
    ) + ".txt"
    with open(output_file_basepath, "w") as text_fout:
        for generated_sentence, ground_true_sentence in zip(sentences, gt_sentences):
            text_fout.write(json.dumps([generated_sentence, ground_true_sentence]) + "\n")

        print(f"written the decoded output to {output_file_basepath}")

    output_file_basepath = os.path.join(
        model_dir,
        f"{model_base_name}.samples_{num_samples}.steps-{args.diffusion_steps}.clamp-{args.clamp}.raw-output-ids-{comments}",
    ) + ".txt"
    with open(output_file_basepath, "w") as text_fout:
        for generated_sentence, ground_true_sentence in zip(raw_sentences, raw_gt_sentences):
            text_fout.write(json.dumps([generated_sentence.tolist(), ground_true_sentence.tolist()]) + "\n")

        print(f"written the decoded output to {output_file_basepath}")


if __name__ == "__main__":
    main()
