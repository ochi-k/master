import cv2
import numpy as np
from tqdm import tqdm


U = [175, 200, 225, 250]


def bg_make():
    for u in U:
        bg_img = 0

        for i in tqdm(range(1001)):
            file_path = f"../data/raw/{u}/u_{u}_q_215_d_18_ppm_0_CR600x2 1836-ST-C-086_{i:08}.bmp"
            img = cv2.imread(file_path, 0)

            if i == 0:
                bg_img = img
            else:
                bg_img = np.minimum(bg_img, img)

        cv2.imwrite(f"../data/bg/{u}.bmp", bg_img)
    print("bg_make fin.\n")


def bg_sub():
    for u in U:
        bg_img = cv2.imread(f"../data/bg/{u}.bmp", 0)

        for i in tqdm(range(1001)):
            file_path = f"../data/raw/{u}/u_{u}_q_215_d_18_ppm_0_CR600x2 1836-ST-C-086_{i:08}.bmp"
            img = cv2.imread(file_path, 0)
            sub_img = img - bg_img
            cv2.imwrite(f"../data/bg_sub/{u}/{i:08}.bmp", sub_img)
    print("bg_sub fin.\n")


def main():
    bg_make()
    bg_sub()


if __name__ == '__main__':
    main()
