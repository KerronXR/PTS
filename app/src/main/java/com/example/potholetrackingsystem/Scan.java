package com.example.potholetrackingsystem;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.provider.MediaStore;
import android.view.Menu;
import android.widget.Toast;
import androidx.core.app.ActivityCompat;
import org.opencv.android.*;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.core.*;


import android.os.Bundle;
import android.util.Log;
import android.view.MenuItem;
import android.view.SurfaceView;
import android.view.WindowManager;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.TensorOperator;
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import static org.opencv.imgproc.Imgproc.*;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class Scan extends CameraActivity implements CvCameraViewListener2 {

    private static final String TAG = "OCVSample::Activity";
    private CameraBridgeViewBase mOpenCvCameraView;

    OpenCVUtils UtilsM;
    LocationFinder locationFinder;
    ImageUploader imageUploader;
    List<String> permissionsList = new ArrayList<>();
    private boolean allSet = false;
    private long sendTime = 0;
    private Mat frame;
    private Mat fixed_frame;
    private Mat grey_frame;
    private Mat grey_copy;
    private Mat canny_image;
    private Mat cropped_frame;
    private Mat cut_frame;
    private Mat[] roi_frame = new Mat[2];
    private Mat frame_copy;
    private Mat result_frame;
    private final Size contour_size = new Size(128, 128);
    private double[][] region_of_interest_vertices;
    private int imageSizeX;
    private int imageSizeY;
    private TensorImage inputImageBuffer;
    private TensorBuffer outputProbabilityBuffer;
    private TensorProcessor probabilityProcessor;
    protected Interpreter interpreter;
    TensorBuffer probabilityBuffer = TensorBuffer.createFixedSize(new int[]{1, 1001}, DataType.UINT8);
    List<MatOfPoint> roi_vertices;

    private final BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            if (status == LoaderCallbackInterface.SUCCESS) {
                Log.i(TAG, "OpenCV loaded successfully");
                mOpenCvCameraView.enableView();
            } else {
                super.onManagerConnected(status);
                Log.i(TAG, "Something went wrong");
            }
        }
    };

    public Scan() {
        UtilsM = new OpenCVUtils();
        region_of_interest_vertices = new double[][]{{100, 720}, {1280 / 2.0, 720 / 2.0}, {1280 - 100, 720}};
        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    protected TensorOperator getPostprocessNormalizeOp() {
        return new NormalizeOp(0.0f, 1.0f);
    }

    /**
     * Called when the activity is first created.
     */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        if (!OpenCVLoader.initDebug())
            Log.e("OpenCv", "Unable to load OpenCV");
        else
            Log.d("OpenCv", "OpenCV loaded");
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.activity_scan);
        locationFinder = new LocationFinder(this);
        imageUploader = new ImageUploader(this);
        try {
            interpreter = new Interpreter(loadModelfile(), null);
        } catch (IOException e) {
            e.printStackTrace();
        }
        // Reads type and shape of input and output tensors, respectively.
        int imageTensorIndex = 0;
        int[] imageShape = interpreter.getInputTensor(imageTensorIndex).shape(); // {1, height, width, 3}
        imageSizeY = imageShape[1];
        imageSizeX = imageShape[2];
        DataType imageDataType = interpreter.getInputTensor(imageTensorIndex).dataType();
        int probabilityTensorIndex = 0;
        int[] probabilityShape =
                interpreter.getOutputTensor(probabilityTensorIndex).shape(); // {1, NUM_CLASSES}
        DataType probabilityDataType = interpreter.getOutputTensor(probabilityTensorIndex).dataType();

        // Creates the input tensor.
        inputImageBuffer = new TensorImage(imageDataType);

        // Creates the output tensor and its processor.
        outputProbabilityBuffer = TensorBuffer.createFixedSize(probabilityShape, probabilityDataType);

        // Creates the post processor for the output probability.
        probabilityProcessor = new TensorProcessor.Builder().add(getPostprocessNormalizeOp()).build();

        if (ActivityCompat.checkSelfPermission(Scan.this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            allSet = false;
            permissionsList.add(Manifest.permission.CAMERA);
        }
        if (ActivityCompat.checkSelfPermission(Scan.this, Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED) {
            allSet = false;
            permissionsList.add(Manifest.permission.ACCESS_FINE_LOCATION);
        }
        if (ActivityCompat.checkSelfPermission(Scan.this, Manifest.permission.ACCESS_COARSE_LOCATION) != PackageManager.PERMISSION_GRANTED) {
            allSet = false;
            permissionsList.add(Manifest.permission.ACCESS_COARSE_LOCATION);
        } else {
            allSet = true;
        }
        if (!allSet) {
            String[] permissionsListToAsk = new String[permissionsList.size()];
            for (int i = 0; i < permissionsList.size(); i++) {
                permissionsListToAsk[i] = permissionsList.get(i);
            }
            ActivityCompat.requestPermissions(Scan.this, permissionsListToAsk, 0);
        }

        mOpenCvCameraView = findViewById(R.id.OpenCv_view);
        mOpenCvCameraView.setMaxFrameSize(1280, 720);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
    }


    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        return true;
    }

    @Override
    protected List<? extends CameraBridgeViewBase> getCameraViewList() {
        return Collections.singletonList(mOpenCvCameraView);
    }

    public void onCameraViewStarted(int width, int height) {
        region_of_interest_vertices[0] = new double[]{width, height};
        region_of_interest_vertices[1] = new double[]{width / 2.0, height / 1.8};
        region_of_interest_vertices[2] = new double[]{0, height};
        roi_vertices = UtilsM.elementToPoints(region_of_interest_vertices);
        Log.e("WIDTH_HEIGHT", width + " " + height);
        frame = new Mat(height, width, CvType.CV_8UC4);
        fixed_frame = new Mat(height, width, CvType.CV_8UC4);
        grey_frame = new Mat(height, width, CvType.CV_8UC1);
        grey_copy = new Mat(height, width, CvType.CV_8UC1);
        canny_image = new Mat(height, width, CvType.CV_8UC4);
        cropped_frame = new Mat(height, width, CvType.CV_8UC4);
        cut_frame = new Mat(height, width, CvType.CV_8UC4);
        roi_frame = new Mat[2];
        roi_frame[0] = new Mat(height, width, CvType.CV_8UC4);
        roi_frame[1] = new Mat(height, width, CvType.CV_8UC4);
        frame_copy = new Mat(height, width, CvType.CV_8UC4);
        result_frame = new Mat(height, width, CvType.CV_8UC4);
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
        if (allSet) {
            frame = inputFrame.rgba();
            fixed_frame = frame.t();
            Core.flip(frame.t(), fixed_frame, 1);
            resize(fixed_frame, fixed_frame, frame.size());
            cvtColor(fixed_frame, grey_frame, COLOR_RGBA2GRAY, 1);
            frame_copy = fixed_frame.clone();
            grey_copy = grey_frame.clone();
            Canny(grey_frame, canny_image, 100, 150);
            cropped_frame = UtilsM.region_of_interest(canny_image, roi_vertices);
            roi_frame = UtilsM.find_road_area(cropped_frame, fixed_frame, grey_copy);
            result_frame = detect_holes(roi_frame[0], roi_frame[1], frame_copy);
            return result_frame;
        }
        else return inputFrame.rgba();
    }

    public boolean onOptionsItemSelected(MenuItem item) {
        return true;
    }

    public void onCameraViewStopped() {
        frame.release();
        fixed_frame.release();
        grey_frame.release();
        grey_copy.release();
        canny_image.release();
        cropped_frame.release();
        cut_frame.release();
        roi_frame[0].release();
        roi_frame[1].release();
        frame_copy.release();
        result_frame.release();
    }

    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    private MappedByteBuffer loadModelfile() throws IOException {
        AssetFileDescriptor assetFileDescriptor = this.getAssets().openFd("PotHoleDetector_128x128_.tflite");
        FileInputStream fileInputStream = new FileInputStream(assetFileDescriptor.getFileDescriptor());
        FileChannel fileChannel = fileInputStream.getChannel();
        long startOffset = assetFileDescriptor.getStartOffset();
        long length = assetFileDescriptor.getLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, length);
    }

    public Mat detect_holes(Mat grey_frame, Mat original_frame, Mat frame_copy) {
        Mat thresh = new Mat();
        Mat image_found = new Mat(contour_size, CvType.CV_8U, Scalar.all(0));
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        double perimeter;
        adaptiveThreshold(grey_frame, thresh, 1, 0, 0, 61, 10);
        findContours(thresh, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);
        Rect rect = null;
        for (int x = 0; x < contours.size(); x++) {
            perimeter = arcLength(new MatOfPoint2f(contours.get(x).toArray()), true);
            Scalar color = new Scalar(0, 0, 255);
            if ((100 < perimeter) && (perimeter < 300)) {
                drawContours(original_frame, contours, x, color, 2, LINE_8, hierarchy, 0, new Point());
                rect = boundingRect(contours.get(x));
                image_found = frame_copy.submat(rect);
                resize(image_found, image_found, contour_size);
                Bitmap bitmap_found = Bitmap.createBitmap(128, 128, Bitmap.Config.ARGB_8888);
                Utils.matToBitmap(image_found, bitmap_found);
                inputImageBuffer.load(bitmap_found);
                interpreter.run(inputImageBuffer.getBuffer(), outputProbabilityBuffer.getBuffer().rewind());
                float[] resultArray = outputProbabilityBuffer.getFloatArray();
                if (resultArray[1] > 0.5) {
                    rectangle(original_frame, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height),
                            new Scalar(255, 0, 0), 5);
                    long currentTime = System.currentTimeMillis();
                    if (currentTime - sendTime > 5000) {
                        Log.e("Location: ", " Lat/Long" + Arrays.toString(locationFinder.getLocation()));
                        imageUploader.uploadImage(locationFinder.getLocation(), bitmap_found);
                        sendTime = System.currentTimeMillis();
                    }
                    putText(original_frame, "Sending...",
                            new Point(original_frame.size(0) / 3.0, original_frame.size(1) / 2.0), FONT_HERSHEY_SIMPLEX,
                            1.5, new Scalar(0, 255, 0), 4, LINE_AA);
                }
            }
        }
        return original_frame;
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String permissions[], int[] grantResults) {
        boolean isPerpermissionForAllGranted = true;
        if (requestCode == 0) {
            if (grantResults.length > 0 && permissions.length == grantResults.length) {
                for (int i = 0; i < permissions.length; i++) {
                    if (grantResults[i] != PackageManager.PERMISSION_GRANTED) {
                        isPerpermissionForAllGranted = false;
                        Toast.makeText(Scan.this, "Permission Denied, You cannot use road scan.", Toast.LENGTH_LONG).show();
                        break;
                    }
                }
            }
            if (isPerpermissionForAllGranted) {
                Toast.makeText(Scan.this, "Permission Granted, Now you can use road scan .", Toast.LENGTH_LONG).show();
                allSet = true;
            }
        }
    }
}
