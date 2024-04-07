import math 
import torch.nn.functional as F
import torch
from tqdm import tqdm
class Trainer:
    def __init__(
        self,
        device,
        model,
        optimizer,
        scheduler = None
        ):
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.model = model 

    #TODO: Make this into two function each for train and val
    @torch.no_grad()
    def evaluate_loss(self,val_loader,vocab_size):
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        for inputs,labels in val_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            #  flatten the logits
            logits = self.model(inputs)
            logits = logits.view(-1,vocab_size)
            labels = labels.view(-1)
            loss = F.cross_entropy(logits, labels)
            total_loss += loss.item() * inputs.size(0)  # Multiply loss by batch size
            total_samples += inputs.size(0)
        average_loss = total_loss / total_samples  # Calculate average loss
        self.model.train()  # Set model back to training mode
        return average_loss


    def train(self,train_loader,val_loader,tokenizer,vocab,max_steps,epoch,eval_iters):
        # if max_steps is None:
        #     data_len = len(data_loader)
        #     max_steps = math.ceil(data_len/batch_size) * epoch
        train_loss = 0.0
        cur_steps = 0
        print("Started training...")
        for e in tqdm(range(epoch)):
            for inputs,labels in train_loader:
                if cur_steps == max_steps:
                    break
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                #  flatten the logits
                logits = self.model(inputs)
                logits = logits.view(-1,len(vocab))
                labels = labels.view(-1)
                loss = F.cross_entropy(logits, labels)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                if cur_steps % eval_iters == 0:
                    val_loss = self.evaluate_loss(val_loader,len(vocab))
                    train_loss = train_loss/eval_iters
                    print(f"step {cur_steps}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")
                    print(self.model.generate(prompt="He shall not",tokenizer=tokenizer,vocab=vocab,max_new_tokens=100,device=self.device))
                    train_loss = 0.0
                cur_steps += 1
            if cur_steps == max_steps:
                    break
        print("Training completed.")
                

