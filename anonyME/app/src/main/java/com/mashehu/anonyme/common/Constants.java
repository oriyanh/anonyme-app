package com.mashehu.anonyme.common;

import android.Manifest;

import java.io.File;

final public class Constants {
	public static final String BULK_CAPTURE_KEY = "isBulkCapture";
	public static int ANONYME_PERMISSION_REQUEST_CODE = 1993;
	public static String ANYNOME_SHARE_SHORTCUT_ID = "1993";
	public static String[] PERMISSIONS = {Manifest.permission.CAMERA,
			Manifest.permission.WRITE_EXTERNAL_STORAGE};
	public static File CACHE_PATH;
	public static File ASSETS_PATH;
	public static File CAMERA_ROLL_PATH;

	public static final String SP_IS_PROCESSING_KEY = "SP_IS_PROCESSING_IMAGES";

	public static final String INTENT_START_ENGINE = "com.mashehu.anonyme.services.action.START_ENGINE";
	public static final String INTENT_START_PROCESSING_IMAGES = "com.mashehu.anonyme.services.action.START_PROCESSING_IMAGES";

	public static final String EXTRA_ENGINE_ASSETS_PATH = "EXTRA_ENGINE_ASSETS_PATH";
	public static final String EXTRA_ENGINE_OUT_DIR = "EXTRA_ENGINE_OUT_DIR";
	public static final String EXTRA_ENGINE_INPUT_PICS = "EXTRA_ENGINE_INPUT_PICS";
	public static final String EXTRA_ENGINE_NUM_IMAGES = "EXTRA_ENGINE_NUM_IMAGES";

	public static final String NOTIFICATION_CH_ID_PROGRESS = "progress_notifications";
	public static final String NOTIFICATION_CH_NAME_PROGRESS = "Progress Notifications";
	public static final String NOTIFICATION_CH_DESC_PROGRESS= "Send notifications that show the progress per image";

	public static final String ENGINE_MODULE_NAME = "AdvBox.applications.face_recognition_attack.facenet_fr";
	public static final String IMAGE_DIRS_ARGUMENT_KEY = "imageDirs";
}
