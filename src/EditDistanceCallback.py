import torch
from torch.utils.data import TensorDataset, DataLoader
import Levenshtein
from transformers import GenerationConfig, PrinterCallback
class EditDistanceCallback(PrinterCallback):
   #TODO: pass device and seqlen 
    def __init__(self, tokenizer, val_dataset, train_dataset, seq_len, device ):
        self.tokenizer = tokenizer
        self.val_dataset = val_dataset
        self.train_dataset = train_dataset
        self.generation_config = GenerationConfig(
            max_length=seq_len,
        )
        self.device = device
    # this is very slow :(, let's only get 100 examples
    def calculate_edit_distance(self, dataset, dataset_name, model, num_samples=100):
        total_edit_distance = 0
        total = 0
        print(f"Calculating edit distance for {dataset_name}")

        input_ids = torch.tensor(dataset['input_ids'], dtype=torch.long)
        labels = torch.tensor(dataset['labels'], dtype=torch.long)
        tensor_dataset = TensorDataset(input_ids, labels)
        batch_size = 16
        dataloader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=False)

        for batch_idx, (encrypted_tensor_sentences, tensor_sentences) in enumerate(dataloader):
            encrypted_tensor_sentences = encrypted_tensor_sentences.to(self.device)
            tensor_sentences = tensor_sentences.to(self.device)

            with torch.no_grad():
                outputs = model.generate(input_ids=encrypted_tensor_sentences, generation_config=self.generation_config)

            for output, tensor_sentence in zip(outputs, tensor_sentences):
                decoded_pred = self.tokenizer.decode(output, skip_special_tokens=True)
                decoded_target = self.tokenizer.decode(tensor_sentence, skip_special_tokens=True)

                edit_distance = Levenshtein.distance(decoded_pred, decoded_target)
                total_edit_distance += edit_distance
                total += 1

                if total <= 5:
                    print(f"Predicted: {decoded_pred}")
                    print(f"Gold Label: {decoded_target}")
                    print("=" * 80)

                if num_samples and total >= num_samples:
                    break

            if num_samples and total >= num_samples:
                break

        avg_edit_distance = total_edit_distance / total
        print(f"Average Edit Distance on {dataset_name} Set: {avg_edit_distance}\n")

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        model.eval()
        # val_samples = len(self.val_dataset)
        self.calculate_edit_distance(self.val_dataset, 'Validation', model)
        self.calculate_edit_distance(self.train_dataset, 'Training', model)#, num_samples=val_samples)



    # def on_epoch_end_old(self, args, state, control, model=None, **kwargs):
        # model.eval()
        # total_edit_distance = 0
        # total=0
# 
        # for encrypted_tensor_sentence, tensor_sentence in zip(self.val_dataset['input_ids'], self.val_dataset['labels']):
            #convert list to tensor
            # tensor_sentence = torch.tensor(tensor_sentence, dtype=torch.long)
            # encrypted_tensor_sentence = torch.tensor(encrypted_tensor_sentence, dtype=torch.long).unsqueeze(0)
            # encrypted_tensor_sentence = encrypted_tensor_sentence.to(device)
# 
            # with torch.no_grad():
                # outputs = model.generate(input_ids=encrypted_tensor_sentence, generation_config=self.generation_config)
# 
            # decoded_pred = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # decoded_target = self.tokenizer.decode(tensor_sentence, skip_special_tokens=True)
# 
            # edit_distance = Levenshtein.distance(decoded_pred, decoded_target)
            # total_edit_distance += edit_distance
            # total+=1
            # if total <5:
                # print(f"Predicted: {decoded_pred}")
                # print(f"Gold Label: {decoded_target}")
                # print("=" * 80)
# 
# 
        # avg_edit_distance = total_edit_distance / len(self.val_dataset)
        # print(f"Average Edit Distance on Validation Set: {avg_edit_distance}\n")
        # print("Sample 5 examples from validation set and print them")
# 