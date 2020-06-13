import asyncio
import io
import glob
import os
import sys
import time
import uuid
import requests
from urllib.parse import urlparse
from io import BytesIO
from PIL import Image, ImageDraw
from azure.cognitiveservices.vision.face import FaceClient, models
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.face.models import TrainingStatusType, Person, SnapshotObjectType, \
    OperationStatusType
import numpy as np
import json


# set to your own subscription key value
PERSON_GROUP_ID = 'anonyme-blackbox'
def train(client, dataset, person_group_id):
    """

    :param FaceClient client:
    :param dataset:
    :return:
    """
    num_skipped = 0
    num_classes_processed = 0
    classes = os.listdir(dataset)
    validation_set = []
    for cls in classes:
        person = client.person_group_person.create(person_group_id, name=cls)
        class_dir = os.path.join(dataset, cls)
        samples = os.listdir(class_dir)
        val_samples = np.random.choice(samples, 5)
        validation_set.extend([os.path.join(class_dir, sample) for sample in val_samples])
        samples_processed = 0
        for sample in np.random.permutation(samples):
            image = os.path.join(class_dir, sample)
            image_file = open(image, 'r+b')
            try:
                client.person_group_person.add_face_from_stream(person_group_id, person.person_id,
                                                                  image_file, detection_model='detection_02')
                samples_processed += 1
            except models.APIErrorException as e:
                print(f"Problem with image {cls}/{sample}: {e} . Skipping.")
                num_skipped += 1
            except Exception as e:
                print(f"General error ocurred with image {cls}/{sample}: {e} . Skipping.")
                num_skipped += 1

            if samples_processed >= 248:
                break
        num_classes_processed += 1
        print(f"Progress {100.*num_classes_processed/len(classes)}%")
    print(f"Total # of samples skipped in training: {num_skipped}")
    client.person_group.train(person_group_id)
    while True:
        training_status = client.person_group.get_training_status(person_group_id)
        print("Training status: {}.".format(training_status.status))
        print()
        if (training_status.status is TrainingStatusType.succeeded):
            break
        elif (training_status.status is TrainingStatusType.failed):
            sys.exit('Training the person group has failed.')
        time.sleep(5)


def validate(image_path, face_client, group_id=None):

    image = open(image_path, 'r+b')

    # Detect faces
    face_ids = []
    faces = face_client.face.detect_with_stream(image, recognition_model='recognition_02', detection_model='detection_02')
    for face in faces:
        face_ids.append(face.face_id)
    results = face_client.face.identify(face_ids, person_group_id=group_id)
    print('Identifying faces in {}'.format(os.path.basename(image.name)))
    if not results:
        print('No person identified in the person group for faces from {}.'.format(
            os.path.basename(image.name)))
        return 0.
    for person in results:
        print(f'Person for face ID {person.face_id} '
              f'is identified in {os.path.basename(image.name)} with a confidence of {person.candidates[0].confidence}.') # Get topmost confidence score
    return person.candidates[0].confidence

def main(dataset, endpoint, key):
    """

    :param str dataset:
    :param str endpoint:
    :param str key:
    :return:
    """
    face_client = get_face_client(endpoint, key)
    train(face_client, dataset, PERSON_GROUP_ID)

def get_face_client(endpoint, key):
    return FaceClient(endpoint, CognitiveServicesCredentials(key))

if __name__ == '__main__':
    dataset = sys.argv[1]
    endpoint = sys.argv[2]
    key = sys.argv[3]
    main(dataset, endpoint, key)

