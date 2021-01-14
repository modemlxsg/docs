import cv2
import imgaug as ia
import imgaug.augmenters as iaa
import os
import glob


class AugTest:

    def __init__(self):

        self.aug = iaa.Sequential([
            iaa.Pad(percent=(0, (0, 0.5), 0, (0, 0.5))),
            iaa.Resize(size={"height": 32, "width": "keep-aspect-ratio"}),
            iaa.Rotate(rotate=(-10, 10)),
        ])

    def load_img(self):
        img = cv2.imread("./imgs/0.jpg")
        img = img[:, :, ::-1]
        aug_img = self.aug(image=img)
        return img, aug_img


if __name__ == "__main__":
    obj = AugTest()
    for i in range(10):
        img, aug = obj.load_img()
        cv2.imshow(f"{i}", aug[:, :, ::-1])

    cv2.waitKey()
    cv2.destroyAllWindows()
