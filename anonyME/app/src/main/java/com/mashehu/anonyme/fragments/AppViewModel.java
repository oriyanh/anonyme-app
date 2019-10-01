package com.mashehu.anonyme.fragments;
import androidx.lifecycle.ViewModel;

import java.util.ArrayList;

public class AppViewModel extends ViewModel {

	ArrayList<String> imagesToProcess;
	boolean multipleSelectionMode = false;
	boolean bulkCaptureMode = false;
	int currentTab = -1;

	public void setCurrentTab(int currentTab) {
		this.currentTab = currentTab;
	}

	public int getCurrentTab() {
		return currentTab;
	}

	public AppViewModel() {
		imagesToProcess = new ArrayList<>();
	}

	public ArrayList<String> getImagePaths() {
		return imagesToProcess;
	}

	public void addImage(String image) {
		if (!imagesToProcess.contains(image)) {
			if (isMultipleSelectionMode()) {
				imagesToProcess.clear();
			}
			imagesToProcess.add(image);
		}
	}

	public void removeImage(String image) {
		this.imagesToProcess.remove(image);
	}

	public boolean isMultipleSelectionMode() {
		return multipleSelectionMode;
	}

	public void setMultipleSelectionMode(boolean isMultipleSelection) {
		this.multipleSelectionMode = isMultipleSelection;
	}

	public boolean isBulkCaptureMode() {
		return bulkCaptureMode;
	}

	public void setBulkCaptureMode(boolean isBulkCapture) {
		this.bulkCaptureMode = isBulkCapture;
	}
}
