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

    # 遍历文件夹，一个文件夹对应一个类别
    tumor_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证顺序一致
    tumor_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(tumor_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    test_images_path = []  # 存储训练集的所有图片路径
    test_images_label = []  # 存储训练集图片对应索引信息

    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in tumor_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))

        for img_path in images:
            test_images_path.append(img_path)
            test_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for testing.".format(len(test_images_path)))

    plot_image = False
    if plot_image:
        # 绘制每种类别个数柱状图
        plt.bar(range(len(tumor_class)), every_class_num, align='center')
        # 将横坐标0,1,2,3,4替换为相应的类别名称
        plt.xticks(range(len(tumor_class)), tumor_class)
        # 在柱状图上添加数值标签
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        # 设置x坐标
        plt.xlabel('image class')
        # 设置y坐标
        plt.ylabel('number of images')
        # 设置柱状图的标题
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

    # print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)], predict[predict_cla].numpy())
    print_res = "iter: {}   label: {}   pred.class: {}   prob: {:.3}".format(iter,
                                                                             class_indict[str(img_label)],
                                                                             class_indict[str(predict_cla)],
                                                                             predict[predict_cla].numpy())
    plt.title(print_res)
    print(print_res)

    # plt.show()
    return class_indict[str(img_label)], class_indict[str(predict_cla)]


if __name__ == '__main__':
    main()