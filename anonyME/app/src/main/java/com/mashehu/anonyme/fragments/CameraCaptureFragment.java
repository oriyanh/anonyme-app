package com.mashehu.anonyme.fragments;

import android.content.Context;
import android.graphics.Matrix;
import android.graphics.Rect;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraManager;
import android.os.Bundle;

import androidx.activity.OnBackPressedCallback;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.camera.core.CameraControl;
import androidx.camera.core.CameraInfoUnavailableException;
import androidx.camera.core.CameraX;
import androidx.camera.core.FlashMode;
import androidx.camera.core.FocusMeteringAction;
import androidx.camera.core.ImageCapture;
import androidx.camera.core.ImageCaptureConfig;
import androidx.camera.core.MeteringPoint;
import androidx.camera.core.MeteringPointFactory;
import androidx.camera.core.Preview;
import androidx.camera.core.PreviewConfig;
import androidx.camera.core.SensorOrientedMeteringPointFactory;
import androidx.camera.core.UseCase;
import androidx.fragment.app.Fragment;
import androidx.lifecycle.ViewModelProviders;
import androidx.navigation.Navigation;

import android.os.Handler;
import android.util.Log;
import android.util.Size;
import android.view.GestureDetector;
import android.view.LayoutInflater;
import android.view.MotionEvent;
import android.view.ScaleGestureDetector;
import android.view.Surface;
import android.view.TextureView;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.mashehu.anonyme.R;
import com.mashehu.anonyme.fragments.ui.GlideApp;

import java.io.File;
import java.math.BigDecimal;
import java.util.Calendar;
import java.util.concurrent.TimeUnit;

import static androidx.camera.core.CameraX.getCameraControl;
import static com.mashehu.anonyme.common.Constants.CACHE_PATH;

/**
 * A simple {@link Fragment} subclass.
 */
public class CameraCaptureFragment extends Fragment implements View.OnLayoutChangeListener {

    private Preview preview = null;
    private ImageCapture imageCapture = null;
    private CameraX.LensFacing lensFacing = CameraX.LensFacing.BACK;
    private ImageCapture.CaptureMode captureMode = ImageCapture.CaptureMode.MAX_QUALITY;
    private float zoomLevel = 1f;
    // private static boolean isBulkCapture = true; // Default value is false, true for debugging purposes
    private String cameraId;
    private TextureView viewFinder;
    private ScaleGestureDetector zoomGestureDetector;
    private GestureDetector focusGestureDetector;
    private static final String TAG = "anonyme.Capture";
    private AppViewModel viewModel;

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
        viewModel = ViewModelProviders.of(getActivity()).get(AppViewModel.class);
        viewFinder = getView().findViewById(R.id.view_finder);
        requireActivity().getOnBackPressedDispatcher().addCallback(new OnBackPressedCallback(true) {
            @Override
            public void handleOnBackPressed() {
                getActivity().finish();
            }
        });
        bindCameraUseCases();
    }

    /**
     * Safely attempts to bind camera use-cases to life cycle
     * @param useCase - use case to bind to life cycle
     * @return True if use-case was successfully bound to lifecycle, otherwise - False.
     */
    private boolean tryBindToLifeCycle(UseCase useCase) {
        try {
            CameraX.bindToLifecycle(this, useCase);
        } catch (IllegalArgumentException e) {
            Log.e(TAG, e.getMessage());

            assert getContext() != null;
            Toast.makeText(getContext().getApplicationContext(), "Bind too many use cases.",
                    Toast.LENGTH_SHORT)
                    .show();

            return false;
        }
        return true;
    }

    /**
     * Get the display's rotation in degrees
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
        Size scaled = calculatePreviewViewDimens(srcResolution, viewWidth,
                viewHeight, displayRotation);

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
        // two digits floating value to do the scale operation. Otherwise,
        // the result may be scaled not large enough and will have some blank lines on the screen.
        xScale = new BigDecimal(xScale).setScale(2, BigDecimal.ROUND_CEILING).floatValue();
        yScale = new BigDecimal(yScale).setScale(2, BigDecimal.ROUND_CEILING).floatValue();

        // Do corresponding scale to resolve deformation problem
        matrix.postScale(xScale, yScale, centerX, centerY);
        textureView.setTransform(matrix);
    }

    @Override
    public void onLayoutChange(View v, int left, int top, int right, int bottom, int oldLeft,
                               int oldTop, int oldRight, int oldBottom) {
        updateTransform();
    }

    private void bindCameraUseCases() //TODO amit: you can pass `view` as an argument here, you don't need to do `getView() inside this function or any of the other setup functions
    {
        CameraX.unbindAll();
        viewFinder.post(this::initializeCameraPreview);
        initializeImageCapture();
    }

    private void initializeCameraPreview() {
        assert getView() != null;

        // TODO:: Get last lens facing from shared preferences
        PreviewConfig previewConfig = new PreviewConfig.Builder()
                .setLensFacing(lensFacing)
                .build();

        preview = new Preview(previewConfig);

        preview.setOnPreviewOutputUpdateListener(output -> {
            ViewGroup parent = (ViewGroup) viewFinder.getParent();
            parent.removeView(viewFinder);
            parent.addView(viewFinder, 0);
            viewFinder.setSurfaceTexture(output.getSurfaceTexture());
        });

        viewFinder.addOnLayoutChangeListener(this);

        if (!tryBindToLifeCycle(preview)) {
            Log.e(TAG, "Failed to bind camera preview to life cycle");
            preview = null;
            return;
        }

        ImageButton flipCameraButton = getView().findViewById(R.id.flip_camera);
        flipCameraButton.setOnClickListener((View v) -> {
                if (lensFacing == CameraX.LensFacing.BACK)
                {
                    lensFacing = CameraX.LensFacing.FRONT;
                }
                else if (lensFacing == CameraX.LensFacing.FRONT)
                {
                    lensFacing = CameraX.LensFacing.BACK;
                }
                zoomLevel = 1f; // Reset zoom level on flip camera
                bindCameraUseCases();
            });

        if ((zoomGestureDetector == null) || (focusGestureDetector == null))
        {
            ZoomListener zoomListener = new ZoomListener();
            zoomGestureDetector = new ScaleGestureDetector(getContext(), zoomListener);

            FocusListener focusListener = new FocusListener();
            focusGestureDetector = new GestureDetector(getContext(), focusListener);

            viewFinder.setOnTouchListener(new View.OnTouchListener() {
                @Override
                public boolean onTouch(View v, MotionEvent event) {
                    zoomGestureDetector.onTouchEvent(event);
                    if (!zoomGestureDetector.isInProgress())
                    {
                        focusGestureDetector.onTouchEvent(event);
                    }
                    return true;
                }
            });
        }
        updateTransform();
    }

    private void refreshFlashButtonIcon() {
        assert getView() != null;
        ImageButton flashToggle = getView().findViewById(R.id.torch_toggle);
        if (imageCapture != null) {
            flashToggle.setVisibility(View.VISIBLE);
            flashToggle.setOnClickListener((view) ->
            {
                FlashMode flashMode = imageCapture.getFlashMode();
                if (flashMode == FlashMode.ON) {
                    imageCapture.setFlashMode(FlashMode.OFF);
                } else if (flashMode == FlashMode.OFF) {
                    imageCapture.setFlashMode(FlashMode.AUTO);
                } else if (flashMode == FlashMode.AUTO) {
                    imageCapture.setFlashMode(FlashMode.ON);
                }
                refreshFlashButtonIcon();
            });
            FlashMode flashMode = imageCapture.getFlashMode();
            switch (flashMode) {
                case ON:
                    flashToggle.setImageResource(R.drawable.ic_flash_on);
                    break;
                case OFF:
                    flashToggle.setImageResource(R.drawable.ic_flash_off);
                    break;
                case AUTO:
                    flashToggle.setImageResource(R.drawable.ic_flash_auto);
                    break;
            }
        }

        // Should never happen
        else {
            flashToggle.setVisibility(View.GONE);
            flashToggle.setOnClickListener(null);
        }
    }

    void initializeImageCapture() {
        ImageCaptureConfig config =
                new ImageCaptureConfig.Builder()
                        .setLensFacing(lensFacing)
                        .setCaptureMode(captureMode)
                        .setUseCaseEventListener(new UseCase.EventListener() {
                            @Override
                            public void onBind(@NonNull String cameraId) {

                            }

                            @Override
                            public void onUnbind() {

                            }
                        })
                        .build();
        imageCapture = new ImageCapture(config);

        if (!tryBindToLifeCycle(imageCapture)) {
            Log.e(TAG, "Failed to bind camera capture to lifecycle");
            imageCapture = null;
            return;
        }

        assert getActivity() != null;
        ImageButton button = getActivity().findViewById(R.id.take_picture);

        // Keeping this so we can remember how to access this directory in the future.
//        final File dir = new File(Environment.getExternalStoragePublicDirectory(DIRECTORY_PICTURES),
//                getActivity().getString(R.string.app_name));

        if (!CACHE_PATH.exists())
        {
            if (!CACHE_PATH.mkdirs())
            {
                Log.e(TAG, "Failed to create directory");
            }
            else
            {
                Log.d(TAG, "Created directory" + CACHE_PATH.toString());
            }
        }
        else
        {
            Log.d(TAG, "Directory exists, all good :)");
        }

        button.setOnClickListener((view) ->
                {
                    final File imageFile = new File(CACHE_PATH,
                            Calendar.getInstance().getTimeInMillis() + ".jpg"); // is this really a jpg?
                    imageCapture.takePicture(
                            imageFile,
                            new ImageCapture.OnImageSavedListener() {
                                @Override
                                public void onImageSaved(@NonNull File file) {
                                    Log.d(TAG, "Saved image to " + file);
                                    viewModel.addImage(imageFile.getAbsolutePath());
                                    if (!viewModel.isBulkCaptureMode())
                                    {
                                        Navigation.findNavController(view).navigate(
                                                R.id.action_cameraCaptureFragment_to_confirmImagesFragment2,
                                                null);
                                    }
                                    else
                                    {
                                        // Get preview view
                                        assert getView() != null;
                                        ImageView lastCapturePreview = getView().findViewById(R.id.last_capture_preview);

                                        // Display image thumbnail in view
                                        assert getContext() != null;
                                        GlideApp.with(getContext()).load(imageFile).galleryThumbnail().into(lastCapturePreview);
                                    }
                                }

                                @Override
                                public void onError(
                                        @NonNull ImageCapture.ImageCaptureError error,
                                        @NonNull String message,
                                        Throwable cause) {
                                    Log.e(TAG, "Failed to save image.", cause);
                                }
                            });

                    // In bulk capture mode, dims screen for 50 ms for shutter effect
                    assert getView() != null;
                    TextView shutterView = getView().findViewById(R.id.shutter_view);
                    Handler shutterHandler = new Handler();

                    shutterView.setVisibility(View.VISIBLE);
                    shutterHandler.postDelayed(()-> shutterView.setVisibility(View.GONE), 50L);
                });

        refreshFlashButtonIcon();
    }

    private class FocusListener extends GestureDetector.SimpleOnGestureListener {

        @Override
        public boolean onDown(MotionEvent e) {
            CameraControl cameraControl;
            try {
                cameraControl = CameraX.getCameraControl(lensFacing);
            }
            catch (CameraInfoUnavailableException exception)
            {
                Log.e(TAG, "Failed to get camera control " + exception);
                return false;
            }

            // Fetches max digital zoom for phone's camera
            assert getActivity() != null;
            CameraManager cameraManager = (CameraManager)getActivity().getSystemService(
                    Context.CAMERA_SERVICE);

            CameraCharacteristics characteristics;
            Rect curCameraCrop;

            try {
                assert cameraManager != null;
                characteristics = cameraManager.getCameraCharacteristics(
                        cameraId);

                curCameraCrop = characteristics.get(
                        CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE);
            }
            catch (CameraAccessException accessException)
            {
                Log.e(TAG, "Failed to fetch camera characteristics");
                return false;
            }

            MeteringPointFactory factory = new SensorOrientedMeteringPointFactory(
                    curCameraCrop.width(), curCameraCrop.height());
            MeteringPoint point = factory.createPoint(e.getX(), e.getY());
            FocusMeteringAction action = FocusMeteringAction.Builder
                    .from(point, FocusMeteringAction.MeteringMode.AF_ONLY)
                    .setAutoFocusCallback(new FocusMeteringAction.OnAutoFocusListener() {
                        @Override
                        public void onFocusCompleted(boolean isFocusLocked) {
                            Toast toast;
                            if (isFocusLocked)
                            {
                                toast = Toast.makeText(getContext(), "Focus succeeded", Toast.LENGTH_SHORT);
                            }
                            else
                            {
                                toast = Toast.makeText(getContext(), "Focus failed", Toast.LENGTH_SHORT);
                            }
                            toast.show();
                        }
                    })
                    .setAutoCancelDuration(5, TimeUnit.SECONDS)
                    .build();

            cameraControl.startFocusAndMetering(action);
            return super.onSingleTapConfirmed(e);
        }
    }

    private class ZoomListener extends ScaleGestureDetector.SimpleOnScaleGestureListener{
        @Override
        public boolean onScale(ScaleGestureDetector detector) {

            // Fetches max digital zoom for phone's camera
            assert getActivity() != null;
            CameraManager cameraManager = (CameraManager)getActivity().getSystemService(
                    Context.CAMERA_SERVICE);

            CameraCharacteristics characteristics;
            float maxZoomLevel;
            Rect curCameraCrop;

            try {

                assert cameraManager != null;
                characteristics = cameraManager.getCameraCharacteristics(
                        cameraId);

                maxZoomLevel = characteristics.get(
                        CameraCharacteristics.SCALER_AVAILABLE_MAX_DIGITAL_ZOOM);

                curCameraCrop = characteristics.get(
                        CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE);
            }
            catch (CameraAccessException e)
            {
                Log.e(TAG, "Failed to fetch camera characteristics");
                return false;
            }
            catch (NullPointerException e)
            {
                Log.e(TAG, "Failed to fetch camera max zoom");
                return false;
            }

            float delta = 0.05f;

            if (detector.getCurrentSpan() > detector.getPreviousSpan())
            {
                if ((maxZoomLevel - zoomLevel) <= delta)
                {
                    delta = maxZoomLevel - zoomLevel;
                }
                zoomLevel += delta;
            }
            else if (detector.getCurrentSpan() < detector.getPreviousSpan())
            {
                if  ((zoomLevel - delta) < 1f)
                {
                    delta = zoomLevel - 1f;
                }
                zoomLevel = zoomLevel - delta;
            }

            assert curCameraCrop != null;
            int cropWidth = curCameraCrop.width() - Math.round((float)curCameraCrop.width() / zoomLevel);
            int cropHeight = curCameraCrop.height() - Math.round((float)curCameraCrop.height() / zoomLevel);
            Rect zoom = new Rect(cropWidth / 2, cropHeight / 2,
                    curCameraCrop.width() - cropWidth / 2,
                    curCameraCrop.height() - cropHeight / 2);

            preview.zoom(zoom);

            return super.onScale(detector);
        }
    }
}
