using System;
using System.Collections.Generic;
using System.Drawing;
using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.CV.CvEnum;
using System.IO;
using System.Windows;
using System.ComponentModel;
using System.Timers;
using System.Windows.Media.Imaging;
using System.Windows.Threading;
using System.Runtime.CompilerServices;
using Microsoft.Win32;
using Emgu.CV.Face;
using System.Text.RegularExpressions;
using System.Linq;

namespace FaceDetectionAndRecognition
{
    public partial class FaceRecognition : Window, INotifyPropertyChanged
    {
        public event PropertyChangedEventHandler PropertyChanged;
        private VideoCapture _videoCapture;
        private readonly Timer _captureTimer;
        private CascadeClassifier _haarCascade;
        private readonly EigenFaceRecognizer _faceRecognizer;

        private Image<Gray, byte> _detectedFaceImage;

        private readonly List<FaceData> _knownFaces;
        private readonly Bitmap _emptyFaceImage;

        string _faceName;
        public string FaceName
        {
            get { return _faceName; }
            set
            {
                _faceName = value.ToUpper();
                lblFaceName.Dispatcher.Invoke(DispatcherPriority.Normal, new Action(() => { lblFaceName.Content = _faceName; }));
                NotifyPropertyChanged();
            }
        }

        string _distance;
        public string Distance
        {
            get { return _distance; }
            set
            {
                _distance = value;
                lblDistance.Dispatcher.Invoke(DispatcherPriority.Normal, new Action(() => { lblDistance.Content = _distance; }));
                NotifyPropertyChanged();
            }
        }

        string _actionButton;
        public string ActionButton
        {
            get { return _actionButton; }
            set
            {
                _actionButton = value.ToUpper();
                btnStatus.Dispatcher.Invoke(DispatcherPriority.Normal, new Action(() => { btnStatus.Content = _actionButton; }));
                NotifyPropertyChanged();
            }
        }

        public Bitmap CameraCapture
        {
            set
            {
                imgCamera.Dispatcher.Invoke(DispatcherPriority.Normal, new Action(() => { imgCamera.Source = BitmapToImageSource(value); }));
                NotifyPropertyChanged();
            }
        }
        public Bitmap CameraCaptureFace
        {
            set
            {
                imgDetectFace.Dispatcher.Invoke(DispatcherPriority.Normal, new Action(() => { imgDetectFace.Source = BitmapToImageSource(value); }));

                if (FaceName == "-")
                    imgDetectFace.Dispatcher.Invoke(DispatcherPriority.Normal, new Action(() => { imgDetectFace.Opacity = 0.7d; }));
                else
                    imgDetectFace.Dispatcher.Invoke(DispatcherPriority.Normal, new Action(() => { imgDetectFace.Opacity = 1.0d; }));

                NotifyPropertyChanged();
            }
        }

        public FaceRecognition()
        {
            _faceRecognizer = new EigenFaceRecognizer(0, double.PositiveInfinity);

            if (File.Exists("face_recognizer"))
                _faceRecognizer.Read("face_recognizer");

            _knownFaces = new List<FaceData>();
            _emptyFaceImage = CreateEmptyFaceImage();

            InitializeComponent();
            _captureTimer = new Timer() { Interval = Config.TimerResponseValue };
            _captureTimer.Elapsed += CaptureTimer_Elapsed;
        }

        protected virtual void NotifyPropertyChanged([CallerMemberName] string propertyName = "")
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }
        private void Window_Loaded(object sender, RoutedEventArgs e)
        {
            LoadKnownFaces();

            if (_knownFaces.Count > 0)
            {
                //var termCrit = new MCvTermCriteria(facesNames.Count, 0.001);
                var mats = new Mat[_knownFaces.Count];

                for (var index = 0; index < _knownFaces.Count; index++)
                    mats[index] = _knownFaces[index].FaceImage.Mat;
                
                _faceRecognizer.Train(mats, _knownFaces.Select(knownFace => knownFace.FaceId).ToArray());
            }

            _videoCapture = new VideoCapture(Config.ActiveCameraIndex);
            _videoCapture.SetCaptureProperty(CapProp.Fps, 30);
            _videoCapture.SetCaptureProperty(CapProp.FrameHeight, 450);
            _videoCapture.SetCaptureProperty(CapProp.FrameWidth, 370);
            _captureTimer.Start();
        }
        private void CaptureTimer_Elapsed(object sender, ElapsedEventArgs e)
        {
            ProcessFrame();
        }
        private void NewFaceButton_Click(object sender, RoutedEventArgs e)
        {
            AddFace();
        }
        private void OpenVideoFile_Click(object sender, RoutedEventArgs e)
        {
            OpenFileDialog openDialog = new OpenFileDialog();
            if (openDialog.ShowDialog().Value == true)
            {
                _captureTimer.Stop();
                _videoCapture.Dispose();

                _videoCapture = new VideoCapture(openDialog.FileName);
                _captureTimer.Start();
                this.Title = openDialog.FileName;
                return;
            }
        }
        private void StatusButton_Click(object sender, RoutedEventArgs e)
        {
            if (ActionButton == "TRAIN")
            {
                var detectedFaceMat = new Mat[1];
                detectedFaceMat[0] = _detectedFaceImage.Resize(100, 100, Inter.Cubic).Mat;

                var detectedFaceLabel = new int[1];
                detectedFaceLabel[0] = _knownFaces.FirstOrDefault(knowFace => knowFace.PersonName.ToUpper() == FaceName).FaceId;

                _faceRecognizer.Train(detectedFaceMat, detectedFaceLabel);
            }
            else if (ActionButton == "ADD NEW FACE")
            {
                AddFace();
            }
        }
        private void Window_Closing(object sender, CancelEventArgs e)
        {
            _faceRecognizer.Write("face_recognizer");
        }

        public void AddFace()
        {
            if (_detectedFaceImage is null)
            {
                MessageBox.Show("No face detected.");
                return;
            }

            _detectedFaceImage = _detectedFaceImage.Resize(100, 100, Inter.Cubic);
            _detectedFaceImage.Save(Config.FacePhotosPath + "face" + (_knownFaces.Count + 1) + Config.ImageFileExtension);
            StreamWriter writer = new StreamWriter(Config.FaceListTextFile, true);
            var personName = Microsoft.VisualBasic.Interaction.InputBox("Your Name");
            writer.WriteLine(string.Format("face{0}:{1}", (_knownFaces.Count + 1), personName));
            writer.Close();
            LoadKnownFaces();
            MessageBox.Show("Succesfull.");
        }

        public void LoadKnownFaces()
        {
            _haarCascade = new CascadeClassifier("Cascades/haarcascade_frontalface_default.xml");
            _knownFaces.Clear();
            string line;
            var reader = new StreamReader(Config.FaceListTextFile);
            while ((line = reader.ReadLine()) != null)
            {
                var lineParts = line.Split(':');
                var faceInstance = new FaceData
                {
                    FaceImage = new Image<Gray, byte>(Config.FacePhotosPath + lineParts[0] + Config.ImageFileExtension),
                    PersonName = lineParts[1],
                    FaceId = int.Parse(Regex.Match(lineParts[0], @"\d+").Value),
                };
                _knownFaces.Add(faceInstance);
            }
            reader.Close();
        }

        private void ProcessFrame()
        {
            using (var imageFrame = _videoCapture.QueryFrame().ToImage<Bgr, byte>())
            {
                if (imageFrame != null)
                {
                    try
                    {
                        var grayFrame = imageFrame.Convert<Gray, byte>();

                        //MCvAvgComp[][] faces = grayframe.DetectHaarCascade(haarCascade, 1.2, 10, HaarDetectionType.DoCannyPruning, new System.Drawing.Size(20, 20));
                        var detectedFaces = _haarCascade.DetectMultiScale(grayFrame, 1.07, 3, System.Drawing.Size.Empty);

                        if (detectedFaces is null || detectedFaces.Length == 0)
                            NoFaceDetected();

                        foreach (var face in detectedFaces)
                        {
                            imageFrame.Draw(face, new Bgr(200, 200, 200), 1);
                            _detectedFaceImage = imageFrame.Copy(face).Convert<Gray, byte>();

                            if (_knownFaces.Count > 0)
                            {
                                RecognizeFace();
                                break; // It's only processing one face at time
                            }
                            else NewFaceFound(); 
                        }

                    }
                    catch (Exception)
                    {
                        //TODO: log
                    }
                    finally
                    {
                        CameraCapture = imageFrame.ToBitmap();
                    }
                }
            }
        }

        private void RecognizeFace()
        {
            try
            {
                var result = _faceRecognizer.Predict(_detectedFaceImage.Resize(100, 100, Inter.Cubic));
                if (result.Label != -1 && result.Label != 0)
                {
                    FaceName = _knownFaces.FirstOrDefault(knownFace => knownFace.FaceId == result.Label).PersonName;
                    Distance = "Distance:\n" + result.Distance.ToString("0.##");
                    CameraCaptureFace = _detectedFaceImage.ToBitmap();
                    ActionButton = "Train";
                }
                else NewFaceFound();
            }
            catch (Exception) { }
        }

        private void NewFaceFound()
        {
            ActionButton = "Add new face";
            FaceName = "-";
        }

        private void NoFaceDetected()
        {
            ActionButton = "No face detected";
            FaceName = "-";
            Distance = "";
            CameraCaptureFace = _emptyFaceImage;
            return;
        }

        public Bitmap CreateEmptyFaceImage()
        {
            var bitmap = new Bitmap(100, 100);
            using (var graphics = Graphics.FromImage(bitmap))
                graphics.Clear(Color.FromArgb(128, 128, 128, 128));
            
            return bitmap;
        }

        /// <summary>
        /// Convert bitmap to bitmap image for image control
        /// </summary>
        /// <param name="bitmap">Bitmap image</param>
        /// <returns>Image Source</returns>
        private BitmapImage BitmapToImageSource(Bitmap bitmap)
        {
            using (var memory = new MemoryStream())
            {
                bitmap.Save(memory, System.Drawing.Imaging.ImageFormat.Bmp);
                memory.Position = 0;
                var bitmapimage = new BitmapImage();
                bitmapimage.BeginInit();
                bitmapimage.StreamSource = memory;
                bitmapimage.CacheOption = BitmapCacheOption.OnLoad;
                bitmapimage.EndInit();

                return bitmapimage;
            }
        }
    }
}