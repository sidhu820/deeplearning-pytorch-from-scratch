import torch
import torchvision
from timeit import default_timer as timer
import torch.nn.utils.prune as prune
import os

def calculate_time(start:float, end : float, device : torch.device = None):
    from timeit import default_timer as timer
    total_time = end - start
    print(f' Evaluation time :{total_time:.3f}')


def print_size_of_model(model):
    torch.save(model.state_dict(), "temp_delme.p")
    print('size (KB) :',os.path.getsize("temp_delme.p")/1e3)
    os.remove('temp_delme.p')


def prune_model(model, pruning_rate=0.3):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):

            # Applying unstructured L1 norm pruning
            prune.l1_unstructured(module, name='weight', amount=pruning_rate)

            prune.remove(module, 'weight')


def count_nonzero_params(model):
    non_zero_count = 0
    zero_count = 0
    for param in model.parameters():
        non_zero_count += torch.count_nonzero(param).item()
    print("The number of non-zero parameters :", non_zero_count)
    for param in model.parameters():
        zero_count += torch.sum(param == 0).item()
    print("The number of zero parameters :", zero_count)


def slice_dataloader(dataloader, start, end):
    sliced_data = []
    current_index = 0
    for inputs, labels in dataloader:
        batch_size = inputs.size(0)
        if current_index + batch_size > start:
            # Find the start index within the current batch
            start_idx = max(start - current_index, 0)
            # Find the end index within the current batch
            end_idx = min(end - current_index, batch_size)
            sliced_inputs = inputs[start_idx:end_idx]
            sliced_labels = labels[start_idx:end_idx]
            sliced_data.append((sliced_inputs, sliced_labels))
            if current_index + batch_size >= end:
                break
        current_index += batch_size
    return sliced_data


def train_with_pruning(model, dataloader, epochs, device):
  criterion = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
  epochs = epochs
  for epoch in range(epochs):
      running_loss = 0.0
      model.train()
      for i, data in enumerate(dataloader, 0):
          inputs, labels = data
          inputs, labels = inputs.to(device), labels.to(device)
          optimizer.zero_grad()
          model = model.to(device)
          outputs = model(inputs)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()
          running_loss += loss.item()

      print(f'Epoch {epoch+1}, Loss: {running_loss/len(dataloader)}')

      if (epoch + 1) % 5 == 0:
          print(f'Pruning after epoch {epoch + 1}')
          prune_model(model, pruning_rate=0.1)
          print('Pruning done.')


def train(model, dataloader, epochs, device):
  criterion = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
  epochs = epochs
  for epoch in range(epochs):
      running_loss = 0.0
      model.train()
      for i, data in enumerate(dataloader, 0):
          inputs, labels = data
          inputs, labels = inputs.to(device), labels.to(device)
          optimizer.zero_grad()
          model = model.to(device)
          outputs = model(inputs)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()
          running_loss += loss.item()

      print(f'Epoch {epoch+1}, Loss: {running_loss/len(dataloader)}')


def test(model, dataloader, device):
  start = timer()
  correct = 0
  total = 0
  model.eval()
  with torch.no_grad():
      for data in dataloader:
          inputs, labels = data
          inputs, labels = inputs.to(device), labels.to(device)
          model = model.to(device)
          outputs = model(inputs)
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()
      print(f'Accuracy of the network on the {total} test images: %d %%' % (100 * correct / total))
  end = timer()
  calculate_time(start, end, device)

        
def caliberate(model,dataloader,device):

    model.eval()
    with torch.no_grad():
        for data in dataloader:
            img, _ =  data
            img = img.to(device)
            model = model.to(device)
            _ = model(img)

