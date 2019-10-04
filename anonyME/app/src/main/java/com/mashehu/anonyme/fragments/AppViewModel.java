package com.mashehu.anonyme.fragments;

import androidx.annotation.NonNull;
import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;
import androidx.lifecycle.ViewModel;

import com.mashehu.anonyme.fragments.ui.RecyclerUtils;

import java.util.ArrayList;

public class AppViewModel extends ViewModel {

	private MutableLiveData<ArrayList<RecyclerUtils.ImageData>> imagesToProcess;
	private ArrayList<RecyclerUtils.ImageData> imagePaths;
	private MutableLiveData<Boolean> multipleSelectionMode;
	private boolean bulkCaptureMode = false;
	private int currentTab = -1;

	public AppViewModel() {
		imagePaths = new ArrayList<>();
		imagesToProcess = new MutableLiveData<>(imagePaths);
		imagesToProcess.observeForever(imageData -> imagePaths = imageData);
		multipleSelectionMode = new MutableLiveData<>(false);
	}

	private MutableLiveData<Boolean> isPagingEnabled;

	public LiveData<Boolean> getPagingEnabled() {
		if (isPagingEnabled == null) {
			isPagingEnabled = new MutableLiveData<>();
			isPagingEnabled.setValue(true);
		}
		return isPagingEnabled;
	}

	public void setPagingEnabled(boolean isPagingEnabled) {
		if (this.isPagingEnabled == null) {
			this.isPagingEnabled = new MutableLiveData<>();
		}

		this.isPagingEnabled.setValue(isPagingEnabled);
	}

	public void setCurrentTab(int currentTab) {
		this.currentTab = currentTab;
	}

	public int getCurrentTab() {
		return currentTab;
	}

	@NonNull
	public LiveData<ArrayList<RecyclerUtils.ImageData>> getImages() {
		return imagesToProcess;
	}

	@NonNull
	public ArrayList<String> getImagePaths() {
		ArrayList<String> paths = new ArrayList<>();
		for (int i = 0; i < imagePaths.size(); i++) {
			paths.add(imagePaths.get(i).getImagePath());
		}
		return paths;
	}

	public void addImage(String image) {
		ArrayList<String> paths = getImagePaths();
		ArrayList<RecyclerUtils.ImageData> images = getImages().getValue();
		if (!paths.contains(image)) {
			if (!isMultipleSelectionMode()) {
				paths.clear();
			}
			RecyclerUtils.ImageData imageData = new RecyclerUtils.ImageData();
			imageData.setImagePath(image);
			images.add(imageData);
		}
		if (images.size() != paths.size()) {
			imagesToProcess.setValue(images);
		}
	}

	public void removeImage(String image) {
		ArrayList<String> paths = getImagePaths();
		paths.remove(image);
		if (paths.size() != getImagePaths().size()) {
			ArrayList<RecyclerUtils.ImageData> images = new ArrayList<>();
			for (String path : paths) {
				RecyclerUtils.ImageData imageData = new RecyclerUtils.ImageData();
				imageData.setImagePath(path);
				images.add(imageData);
			}
			imagesToProcess.setValue(images);
		}
	}


	public boolean isMultipleSelectionMode() {
		try {
			return getMultipleSelectionMode().getValue();
		}
		catch (NullPointerException e) {
			return false;
		}
	}

	@NonNull
	public LiveData<Boolean> getMultipleSelectionMode() {
		if (this.multipleSelectionMode == null) {
			this.multipleSelectionMode = new MutableLiveData<>();
		}

		return multipleSelectionMode;
	}

	public void setMultipleSelectionMode(boolean isMultipleSelection) {
		this.multipleSelectionMode.setValue(isMultipleSelection);
	}

	public boolean isBulkCaptureMode() {
		return bulkCaptureMode;
	}

	public void setBulkCaptureMode(boolean isBulkCapture) {
		this.bulkCaptureMode = isBulkCapture;
	}

	public void clearImages() {
		imagesToProcess.setValue(new ArrayList<>());
	}
}
