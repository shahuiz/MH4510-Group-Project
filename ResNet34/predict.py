import os
import json
import pandas as pd

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import resnet34


def main():
    test_images_path, test_images_label = read_data(root="./data/Test/")
    length = len(test_images_path)

    truth_table = pd.DataFrame(0, columns=['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor'],
                               index=['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor'])

    for i in range(0, length):
        orig_cla, pred_cla = resnet34_classifier(test_images_path[i], test_images_label[i], i)
        truth_table[pred_cla][orig_cla] += 1

    print("\nSummary:\n")
    print(truth_table)


def read_data(root: str):
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    tumor_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    tumor_class.sort()
    class_indices = dict((k, v) for v, k in enumerate(tumor_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    test_images_path = []  
    test_images_label = []  

    every_class_num = []  
    supported = [".jpg", ".JPG", ".png", ".PNG"]  
    
    for cla in tumor_class:
        cla_path = os.path.join(root, cla)
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        image_class = class_indices[cla]
        every_class_num.append(len(images))

        for img_path in images:
            test_images_path.append(img_path)
            test_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for testing.".format(len(test_images_path)))

    plot_image = False
    if plot_image:
        plt.bar(range(len(tumor_class)), every_class_num, align='center')
        plt.xticks(range(len(tumor_class)), tumor_class)
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        plt.xlabel('image class')
        plt.ylabel('number of images')
        plt.title('tumor class distribution')
        plt.show()

    return test_images_path, test_images_label


def resnet34_classifier(img_path: str, img_label: str, iter: int):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    # img_path = "./data/Test/glioma_tumor/image(1).jpg"
    # assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create model
    model = resnet34(num_classes=4).to(device)

    # load model weights
    weights_path = "./weights/1016-0945-16-0.0001-model-23.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # prediction
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()
        
    print_res = "iter: {}   label: {}   pred.class: {}   prob: {:.3}".format(iter,
                                                                             class_indict[str(img_label)],
                                                                             class_indict[str(predict_cla)],
                                                                             predict[predict_cla].numpy())
    plt.title(print_res)
    print(print_res)

    return class_indict[str(img_label)], class_indict[str(predict_cla)]


if __name__ == '__main__':
    main()
