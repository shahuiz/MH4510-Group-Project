import os
import json

import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

from vit_model import vit_large as creat_model_large
from vit_model import vit_base as creat_model_base


def main():
    test_images_path, test_images_label = read_data(root="C:\\Users\\11194\\OneDrive\\Desktop\\archive\\Testing")
    length = len(test_images_path)

    truth_table = pd.DataFrame(0, columns=['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor'],
                               index=['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor'])
    #store the prediction result
    predict = []
    for i in range(0, length):
        orig_cla, pred_cla = vit_classifier(test_images_path[i], test_images_label[i], i)
        if pred_cla == 'glioma_tumor':
            predict.append(0)
        elif pred_cla == 'meningioma_tumor':
            predict.append(1)
        elif pred_cla == 'no_tumor':
            predict.append(2)
        else:
            predict.append(3)
        truth_table[pred_cla][orig_cla] += 1

    print("\nSummary:\n")
    print(truth_table) #show the truth table
    f1_rf = f1_score(test_images_label, predict, average='weighted')  #calculate f1 score
    print(f1_rf)
    #accuracy was calculated from the truth table by hand

def read_data(root: str):
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # read the entire file
    tumor_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # sort to make it matches
    tumor_class.sort()
    # generate categories' names and corresponding index
    class_indices = dict((k, v) for v, k in enumerate(tumor_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    test_images_path = []  # Store all image paths of the training set
    test_images_label = []  # Store the index information corresponding to the training set images

    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # Iterate through the files in each folder
    for cla in tumor_class:
        cla_path = os.path.join(root, cla)
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # Get the index corresponding to the category
        image_class = class_indices[cla]
        # Record the number of samples in the category
        every_class_num.append(len(images))

        for img_path in images:
            test_images_path.append(img_path)
            test_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for testing.".format(len(test_images_path)))

    return test_images_path, test_images_label


def vit_classifier(img_path: str, img_label: str, iter: int):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # data processing
    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    # load image
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)
    img = data_transform(img)
    # expand the batch dimension
    img = torch.unsqueeze(img, dim=0)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create model
    model = creat_model_large(num_classes=4, has_logits=True).to(device)

    # load trained model weights
    model_weight_path = "./weights/modelb-18.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()
    #predict and show each prediction result
    print_res = "iter: {}   label: {}   pred.class: {}   prob: {:.3}".format(iter, class_indict[str(img_label)],
                                                                             class_indict[str(predict_cla)],
                                                                             predict[predict_cla].numpy())
    plt.title(print_res)
    print(print_res)

    return class_indict[str(img_label)], class_indict[str(predict_cla)]


if __name__ == '__main__':
    main()