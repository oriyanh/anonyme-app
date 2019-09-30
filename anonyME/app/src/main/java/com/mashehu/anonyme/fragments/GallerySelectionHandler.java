package com.mashehu.anonyme.fragments;

import android.view.View;

import androidx.annotation.NonNull;

import com.mashehu.anonyme.fragments.ui.ThumbnailViewHolder;

public interface GallerySelectionHandler {

	public void setSelectionMode(boolean multipleSelectionMode);

	public boolean getSelectionMode();

	public void toggleCheckbox(ThumbnailViewHolder holder, String img);

	public void removeImage(String img);

	public void submitImage(String img);

	public void clear();

	public void startProcessing(View v);

}
