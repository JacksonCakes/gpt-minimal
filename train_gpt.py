import torch
from gpt import GPT
from trainer import Trainer
from dataset import CausalLMDataset
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import argparse
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# read .txt data
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

def save_model(model,optimizer,params,args,out_dir="./"):
    checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': args,
                    "num_params": params
                }
    print(f"saving checkpoint to {out_dir}")
    torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
def main(args):
    tokenizer = get_tokenizer("basic_english") # split sentence by space
    # tokenize the text
    tokenized_text = [list(tokenizer(text))]
    # build the vocabulary from the tokenized text
    vocab = build_vocab_from_iterator(tokenized_text)
    # convert to numerical representation
    input_ids = [vocab[token] for token in tokenized_text[0]]
    n = int(args.val_ratio*len(input_ids)) # first 90% will be train, rest val
    train_input_ids = input_ids[:n]
    val_input_ids = input_ids[n:]
    # Create the dataset and dataloader
    train_dataset = CausalLMDataset(train_input_ids, args.block_size)
    val_dataset = CausalLMDataset(val_input_ids, args.block_size)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    head_size = args.num_embeddings // args.num_heads
    model = GPT(
        num_layer = args.num_layer,
        vocab_size=len(vocab),
        num_embeddings = args.num_embeddings,
        num_heads = args.num_heads,
        head_size = head_size,
        block_size = args.block_size,
        hs_scale_factor = args.hs_scale_factor
    )
    model = model.to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    trainer = Trainer(
        args.device,
        model,
        optimizer,
    )
    trainer.train(
        train_loader = train_loader,
        val_loader = val_loader,
        max_steps = args.max_steps,
        tokenizer=tokenizer,
        epoch = args.epoch,
        eval_iters = args.eval_iters,
        vocab = vocab
    )
    save_model(model,optimizer,model.num_parameters(),args)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_layer", type=int,default=2)
    parser.add_argument("--num_embeddings", type=int,default=128)
    parser.add_argument("--num_heads", type=int,default=8)
    parser.add_argument("--block_size", type=int,default=128)
    parser.add_argument("--hs_scale_factor", type=int,default=4)
    parser.add_argument("--device", type=str,default="cuda")
    parser.add_argument("--batch_size", type=int,default=128)
    parser.add_argument("--epoch", type=int,default=10)
    parser.add_argument("--learning_rate", type=float,default=3e-4)
    parser.add_argument("--max_steps", type=int,default=5000)
    parser.add_argument("--val_ratio", type=float,default=0.9)
    parser.add_argument("--eval_iters", type=int,default=500)
    args = parser.parse_args()
    main(args)
