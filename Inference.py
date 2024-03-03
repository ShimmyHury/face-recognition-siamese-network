import cv2
import os
from Encoder import *
import torchvision.transforms as T
import torch.nn.functional as F
import os.path
import argparse


def compute_similarity(encoder, face1, face2, metric='Euclidean'):
    # Getting the encodings for the passed faces
    with torch.no_grad():
        tensor1 = encoder(face1.to(device))
        tensor2 = encoder(face2.to(device))
    if metric == 'Cosine':
        return F.cosine_similarity(tensor1, tensor2)
    else:
        distance = (tensor1 - tensor2).pow(2).sum(1)
        return 0.5 * (2 - distance)


def face_detection(cls, image):
    return cls.detectMultiScale(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 1.3, 5)


def get_preprocessing(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    return T.Compose([T.ToTensor(), T.Normalize(mean=mean, std=std)])


def get_face_for_reference(path, classifier):
    if os.path.isfile('reference.png'):
        ref = cv2.imread('reference.png')
        cv2.imshow(path, ref)
        cv2.waitKey(1)
        while True:
            user_input = input("Use this face for recognition? [Y/N]")
            cv2.destroyAllWindows()
            if user_input.lower() == 'y':
                return ref
            elif user_input.lower() == 'n':
                os.remove('reference.png')
                break
            else:
                continue

    while not os.path.isfile(path):
        _, img = cam.read()
        faces = face_detection(classifier, img)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
            cv2.imshow(path, img)
            cv2.waitKey(1)
            user_input = input("Use this face for recognition? [Y/N]")
            cv2.destroyAllWindows()
            if user_input.lower() == 'y':
                cropped = img[y:y + h, x:x + w]
                resized_cropped = cv2.resize(cropped, (128, 128), interpolation=cv2.INTER_AREA)
                cv2.imwrite(path, resized_cropped)
                ref = resized_cropped
                return ref
            elif user_input.lower() == 'n':
                continue
            else:
                print('Invalid input. Use this face for recognition? [Y/N]')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Inference Arguments",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-m", "--model_path", default='./trained_model', help="Path to model (save/load)")
    parser.add_argument("-c", "--cam_port", default=0, type=int, help="Camera port for acquisition")
    parser.add_argument("-s", "--sim_metric", default='Cosine',
                        help="Similarity metric, options: 'Euclidean', 'Cosine'")
    parser.add_argument("-r", "--ref_path", default='./reference.png', help="Path to reference image (save/load)")
    args = vars(parser.parse_args())

    print(args)

    path_to_model = args['model_path']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = ResNetEncoder().to(device)
    model.eval()
    try:
        model.load_state_dict(torch.load(path_to_model))
        print(f"{path_to_model} valid. state dict loaded")
    except Exception as e:
        print(e)
        print(f"{path_to_model} not valid. state dict not loaded. training from scratch.")

    cam_port = args['cam_port']
    cam = cv2.VideoCapture(cam_port)
    if cam.isOpened():

        transform = get_preprocessing()

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        path_to_ref = args['ref_path']
        reference = get_face_for_reference(path_to_ref, face_cascade)
        while True:
            reference_img = cv2.imread(path_to_ref)
            reference_img = cv2.cvtColor(reference_img, cv2.COLOR_BGR2RGB)
            reference = transform(reference_img).unsqueeze(0)
            result, frame = cam.read()
            if result:
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                normalized_gray = cv2.normalize(gray, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
                                                dtype=cv2.CV_32F)
                faces = face_detection(face_cascade, image_rgb)
                for (x, y, w, h) in faces:
                    cropped_image = image_rgb[y:y + h, x:x + w]
                    resized_cropped_image = cv2.resize(cropped_image, (128, 128), interpolation=cv2.INTER_AREA)
                    face = transform(resized_cropped_image).unsqueeze(0)
                    score = compute_similarity(model, reference, face, metric=args['sim_metric'])
                    if score >= 0.8:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    else:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    org = (x - 1, y - 1)
                    fontScale = 0.5
                    color = (255, 255, 255)
                    thickness = 1
                    text = 'score: ' + "{:.3f}".format(score.item())
                    frame = cv2.putText(frame, text, org, font,
                                        fontScale, color, thickness, cv2.LINE_AA)
            cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('Video', cv2.WND_PROP_VISIBLE) < 1:
                cv2.destroyAllWindows()
                break
    else:
        print('No camera available on this port')
