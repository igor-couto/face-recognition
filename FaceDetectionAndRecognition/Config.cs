﻿namespace FaceDetectionAndRecognition
{
    public static class Config
    {
        public static string HaarCascadePath = "haarcascade_frontalface_default.xml";
        public static string FacePhotosPath = "Source\\Faces\\";
        public static string FaceListTextFile = "Source\\FaceList.txt";
        public static int TimerResponseValue = 500;
        public static string ImageFileExtension = ".bmp";
        public static int ActiveCameraIndex = 0;//0: Default active camera device
    }
}
