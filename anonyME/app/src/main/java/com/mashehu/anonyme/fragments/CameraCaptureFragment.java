package com.mashehu.anonyme.fragments;

import android.Manifest;
import android.app.AlertDialog;
import android.content.DialogInterface;
import android.content.pm.PackageManager;
import android.graphics.Matrix;
import android.os.Bundle;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.camera.core.CameraInfoUnavailableException;
import androidx.camera.core.CameraX;
import androidx.camera.core.Preview;
import androidx.camera.core.PreviewConfig;
import androidx.core.app.ActivityCompat;
import androidx.fragment.app.Fragment;

import android.util.Log;
import android.util.Size;
import android.view.LayoutInflater;
import android.view.Surface;
import android.view.TextureView;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Toast;

import com.mashehu.anonyme.R;
import com.mashehu.anonyme.common.Constants;

import java.math.BigDecimal;

import static androidx.constraintlayout.widget.Constraints.TAG;

/**
 * A simple {@link Fragment} subclass.
 */
public class CameraCaptureFragment extends Fragment implements View.OnLayoutChangeListener{

    private int displayId = -1;
    private Preview preview = null;
    private CameraX.LensFacing lensFacing = CameraX.LensFacing.BACK;
//    private File outputDirectory = null;

    public CameraCaptureFragment() {
        // Required empty public constructor
    }

    @Override
    public void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        this.setRetainInstance(true);
    }

    @Override
    public void onResume() {
        super.onResume();
    }

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                             Bundle savedInstanceState) {
        // Inflate the layout for this fragment
        return inflater.inflate(R.layout.fragment_camera_capture, container, false);
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);

        assert getActivity() != null;
        assert getView() != null;
        TextureView viewFinder = getView().findViewById(R.id.view_finder);

//        DisplayManager displayManager = (DisplayManager)viewFinder.getContext().getSystemService(
//                Context.DISPLAY_SERVICE);
//
//        displayManager.registerDisplayListener(new DisplayManager.DisplayListener() {
//            @Override
//            public void onDisplayAdded(int displayId) {
//                return;
//            }
//
//            @Override
//            public void onDisplayRemoved(int displayId) {
//                return;
//            }
//
//            @Override
//            public void onDisplayChanged(int displayId) {
//                if (displayId == CameraCaptureFragment.this.displayId)
//                {
//                    assert getView() != null;
//                    CameraCaptureFragment.this.preview.setTargetRotation(
//                            getView().getDisplay().getRotation());
//                }
//            }
//        }, null);

//        assert getContext() != null;
//        outputDirectory = MainActivity.getOutputDirectory(getContext());
//
//        viewFinder.post(() -> {
//                displayId = viewFinder.getDisplay().getDisplayId();
//            });

        boolean cameraPermission = ActivityCompat.checkSelfPermission(getActivity(),
                Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED;
        if (!cameraPermission)
        {
            ActivityCompat.requestPermissions(getActivity(),
                    new String[]{Manifest.permission.CAMERA},
                    Constants.CAMERA_PERMISSION_REQUEST_CODE);
        }
        else
        {
            viewFinder.post(this::startCamera);
        }
    }

    /**
     * @return One of 0, 90, 180, 270.
     */
    private int getDisplayRotation() {

        assert getActivity() != null;
        int displayRotation = getActivity().getWindowManager().getDefaultDisplay().getRotation();
        switch (displayRotation) {
            case Surface.ROTATION_0:
                displayRotation = 0;
                break;
            case Surface.ROTATION_90:
                displayRotation = 90;
                break;
            case Surface.ROTATION_180:
                displayRotation = 180;
                break;
            case Surface.ROTATION_270:
                displayRotation = 270;
                break;
            default:
                throw new UnsupportedOperationException(
                        "Unsupported display rotation: " + displayRotation);
        }
        return displayRotation;
    }

    private Size calculatePreviewViewDimens(
            Size srcSize, int parentWidth, int parentHeight, int displayRotation) {
        int inWidth = srcSize.getWidth();
        int inHeight = srcSize.getHeight();
        if (displayRotation == 0 || displayRotation == 180) {
            // Need to reverse the width and height since we're in landscape orientation.
            inWidth = srcSize.getHeight();
            inHeight = srcSize.getWidth();
        }
        int outWidth = parentWidth;
        int outHeight = parentHeight;
        if (inWidth != 0 && inHeight != 0) {
            float vfRatio = inWidth / (float) inHeight;
            float parentRatio = parentWidth / (float) parentHeight;
            // Match shortest sides together.
            if (vfRatio < parentRatio) {
                outWidth = parentWidth;
                outHeight = Math.round(parentWidth / vfRatio);
            } else {
                outWidth = Math.round(parentHeight * vfRatio);
                outHeight = parentHeight;
            }
        }
        return new Size(outWidth, outHeight);
    }

    private void updateTransform() {
        String cameraId = null;
        PreviewConfig config = (PreviewConfig) preview.getUseCaseConfig();
        CameraX.LensFacing previewLensFacing = config.getLensFacing(/*valueIfMissing=*/ null);

        // Make sure lensFacing is coordinated with preview lens facing
        if (previewLensFacing != lensFacing) {
            throw new IllegalStateException(
                    "Invalid preview lens facing: "
                            + previewLensFacing
                            + " Should be: "
                            + lensFacing);
        }

        try {
            cameraId = CameraX.getCameraWithCameraDeviceConfig(config);
        } catch (CameraInfoUnavailableException e) {
            throw new IllegalArgumentException(
                    "Unable to get camera id for the camera device config "
                            + config.getLensFacing(), e);
        }
        Size srcResolution = preview.getAttachedSurfaceResolution(cameraId);
        if (srcResolution.getWidth() == 0 || srcResolution.getHeight() == 0) {
            return;
        }

        assert getActivity() != null;
        TextureView textureView = getActivity().findViewById(R.id.view_finder);
        if (textureView.getWidth() == 0 || textureView.getHeight() == 0) {
            return;
        }

        Matrix matrix = new Matrix();
        int left = textureView.getLeft();
        int right = textureView.getRight();
        int top = textureView.getTop();
        int bottom = textureView.getBottom();

        // Compute the preview ui size based on the available width, height, and ui orientation.
        int viewWidth = (right - left);
        int viewHeight = (bottom - top);

        int displayRotation = getDisplayRotation();
        Size scaled =
                calculatePreviewViewDimens(
                        srcResolution, viewWidth, viewHeight, displayRotation);
        // Compute the center of the view.
        int centerX = viewWidth / 2;
        int centerY = viewHeight / 2;
        // Do corresponding rotation to correct the preview direction
        matrix.postRotate(-getDisplayRotation(), centerX, centerY);
        // Compute the scale value for center crop mode
        float xScale = scaled.getWidth() / (float) viewWidth;
        float yScale = scaled.getHeight() / (float) viewHeight;
        if (getDisplayRotation() == 90 || getDisplayRotation() == 270) {
            xScale = scaled.getWidth() / (float) viewHeight;
            yScale = scaled.getHeight() / (float) viewWidth;
        }
        // Only two digits after the decimal point are valid for postScale. Need to get ceiling of
        // two
        // digits floating value to do the scale operation. Otherwise, the result may be scaled not
        // large enough and will have some blank lines on the screen.
        xScale = new BigDecimal(xScale).setScale(2, BigDecimal.ROUND_CEILING).floatValue();
        yScale = new BigDecimal(yScale).setScale(2, BigDecimal.ROUND_CEILING).floatValue();
        // Do corresponding scale to resolve the deformation problem
        matrix.postScale(xScale, yScale, centerX, centerY);
        // Compute the new left/top positions to do translate
        // TODO:: dafuq is this shit
//        int layoutL = centerX - (scaled.getWidth() / 2);
//        int layoutT = centerY - (scaled.getHeight() / 2);
        textureView.setTransform(matrix);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (grantResults[0] == PackageManager.PERMISSION_GRANTED)
        {
            startCamera();
        }
        else
        {
            if (getActivity() != null)
            {
                if (ActivityCompat.shouldShowRequestPermissionRationale(getActivity(),
                        Manifest.permission.CAMERA))
                {
                    AlertDialog.Builder askPermissionsDialog = new AlertDialog.Builder(getActivity());
                    askPermissionsDialog.setMessage("Pretty please?").setPositiveButton("Yes",
                            new DialogInterface.OnClickListener() {
                        @Override
                        public void onClick(DialogInterface dialog, int which) {
                            if (getActivity() != null)
                            {
                                ActivityCompat.requestPermissions(getActivity(),
                                        new String[]{Manifest.permission.CAMERA},
                                        Constants.CAMERA_PERMISSION_REQUEST_CODE);
                            }
                        }
                    }).setNegativeButton("No", new DialogInterface.OnClickListener() {
                        @Override
                        public void onClick(DialogInterface dialog, int which) {
                            if (getActivity() != null)
                            {
                                getActivity().finish();
                            }
                        }
                    });
                    askPermissionsDialog.create();
                }
            }
        }
    }

    private void startCamera()
    {
        assert getView() != null;
        TextureView viewFinder = getView().findViewById(R.id.view_finder);
//
//        assert getContext() != null;
//        DisplayMetrics metrics = getContext().getResources().getDisplayMetrics();
//
//        assert getActivity() != null;
//        getActivity().getWindowManager().getDefaultDisplay().getRealMetrics(metrics);
//        Rational screenApectRatio = new Rational(metrics.widthPixels, metrics.heightPixels);

        // TODO:: Get last lens facing from shared preferences
        PreviewConfig previewConfig = new PreviewConfig.Builder()
                .setLensFacing(lensFacing)
//                .setTargetAspectRatio(screenApectRatio)
//                .setTargetRotation(getActivity().getWindowManager().getDefaultDisplay()
//                        .getRotation())
                .build();

        preview = new Preview(previewConfig);
        preview.setOnPreviewOutputUpdateListener(output -> {
                ViewGroup parent = (ViewGroup)viewFinder.getParent();
                parent.removeView(viewFinder);
                parent.addView(viewFinder, 0);
                viewFinder.setSurfaceTexture(output.getSurfaceTexture());
        });

//        ImageCaptureConfig imageCaptureConfig = new ImageCaptureConfig.Builder()
//                .setTargetAspectRatio(screenApectRatio)
//                .setCaptureMode(ImageCapture.CaptureMode.MIN_LATENCY)
//                .build();

//        ImageCapture imageCapture = new ImageCapture(imageCaptureConfig);
        viewFinder.addOnLayoutChangeListener(this);

        try {
            CameraX.bindToLifecycle(this, preview);
        } catch (IllegalArgumentException e) {
            Log.e(TAG, e.getMessage());

            assert getContext() != null;
            Toast.makeText(getContext().getApplicationContext(), "Bind too many use cases.",
                    Toast.LENGTH_SHORT)
                    .show();

            preview = null;
            return;
        }

        updateTransform();
    }

    @Override
    public void onLayoutChange(View v, int left, int top, int right, int bottom, int oldLeft, int oldTop, int oldRight, int oldBottom) {
        updateTransform();
    }
}
