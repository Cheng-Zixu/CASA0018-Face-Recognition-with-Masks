import dlib
import numpy as np

import cv2
import os

def generate_embedding(user_face_dir):

	landmark = dlib.shape_predictor(r'./shape_predictor_68_face_landmarks.dat')
	recognition = dlib.face_recognition_model_v1('./dlib_face_recognition_resnet_model_v1.dat')
	face = ""
	for filepath,dirnames,filenames in os.walk(user_face_dir):
		face = filenames[0]
	im_face = cv2.imread(user_face_dir + '/' + face)
	im_face = cv2.cvtColor(im_face, cv2.COLOR_BGR2RGB)
	rect = dlib.rectangle(0, 0, int(im_face.shape[0]), int(im_face.shape[1]))
	# print(im_face.shape)
	shape = landmark(im_face, rect)
	face = dlib.get_face_chip(im_face, shape, size=150)
	face_embedding = recognition.compute_face_descriptor(face)

	return face_embedding
    

def check_embedding(face_dir, face_embedding_dir, type):
	if (type == 0):
		npy_name = '_mask.npy'
	elif (type == 1):
		npy_name = '_nomask.npy'

	for filepath,dirnames,filenames in os.walk(face_dir):
		for dirname in dirnames:
			embedding_dir = face_embedding_dir + '/' + dirname
			if(os.path.exists(embedding_dir) and os.path.isfile(embedding_dir + '/' + dirname + npy_name)):
				print(dirname + npy_name + " exists.")
			else:
				embedding = np.array(generate_embedding(mask_face_dir + '/' + dirname))
				np.save(embedding_dir + '/' + dirname + npy_name, vec1)

				print(dirname + npy_name + " has been generated.")

def main(face_dir, face_embedding_dir):

	# generate mask face embedding
	mask_face_dir = face_dir + '/mask'
	mask_face_embedding_dir = face_embedding_dir + '/mask'
	check_embedding(mask_face_dir, mask_face_embedding_dir, 0)

	# generate nomask face embedding
	nomask_face_dir = face_dir + '/nomask'
	nomask_face_embedding_dir = face_embedding_dir + '/nomask'
	check_embedding(nomask_face_dir, nomask_face_embedding_dir, 1)
	print("Face embedding updated!")

if __name__ == "__main__":
	face_dir = './yolov5/Face'
	face_embedding_dir = './yolov5/FaceEmbedding'
	main(face_dir, face_embedding_dir)