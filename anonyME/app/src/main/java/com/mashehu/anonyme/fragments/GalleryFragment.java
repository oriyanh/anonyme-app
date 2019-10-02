package com.mashehu.anonyme.fragments;

import android.os.Bundle;

import androidx.activity.OnBackPressedCallback;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;
import androidx.lifecycle.ViewModelProviders;
import androidx.navigation.Navigation;
import androidx.recyclerview.widget.GridLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;

import com.mashehu.anonyme.R;
import com.mashehu.anonyme.common.Utilities;
import com.mashehu.anonyme.fragments.ui.ImageData;
import com.mashehu.anonyme.fragments.ui.ThumbnailAdapter;
import com.mashehu.anonyme.fragments.ui.ThumbnailViewHolder;

import java.util.ArrayList;

import static com.mashehu.anonyme.common.Constants.IMAGE_DIRS_ARGUMENT_KEY;

public class GalleryFragment extends Fragment implements GallerySelectionHandler {
	public static final String TAG = "anonyme.GalleryFragment";
	RecyclerView recyclerView;
	GridLayoutManager layoutManager;
	boolean selectionMode;
	Button sendImagesBtn;
	AppViewModel viewModel;

	@Override
	public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container,
							 @Nullable Bundle savedInstanceState) {
		return inflater.inflate(R.layout.fragment_gallery, container, false);
	}

	@Override
	public void onActivityCreated(@Nullable Bundle savedInstanceState) {
		super.onActivityCreated(savedInstanceState);

		assert getActivity() != null;
		viewModel = ViewModelProviders.of(getActivity()).get(AppViewModel.class);

		sendImagesBtn = getActivity().findViewById(R.id.sendImageBtn);

		sendImagesBtn.setOnClickListener(this::startProcessing);

		recyclerView = getActivity().findViewById(R.id.galleryRecyclerView);
		layoutManager = new GridLayoutManager(getActivity().getApplicationContext(), 4);
		recyclerView.setLayoutManager(layoutManager);

		ArrayList<ImageData> images = Utilities.getGalleryContent();
		ThumbnailAdapter adapter = new ThumbnailAdapter(getActivity().getApplicationContext(), this, images);
		recyclerView.setAdapter(adapter);
		requireActivity().getOnBackPressedDispatcher().addCallback(new OnBackPressedCallback(true) {
			@Override
			public void handleOnBackPressed() {
				getActivity().finish();
			}
		});
	}

	public void setSelectionMode(boolean isMultipleSelection) {
		if (isMultipleSelection == this.selectionMode) {
			return;
		}
		viewModel.setMultipleSelectionMode(isMultipleSelection);
		this.selectionMode = isMultipleSelection;
	}

	public boolean getSelectionMode() {
		return selectionMode;
	}

	@Override
	public void toggleCheckbox(ThumbnailViewHolder holder, String img) {
		if (viewModel.isMultipleSelectionMode()) {
			if (holder.checkbox.getVisibility() == View.VISIBLE) {
				holder.checkbox.setVisibility(View.INVISIBLE);
				removeImage(img);
			}
			else {
				holder.checkbox.setVisibility(View.VISIBLE);
				submitImage(img);
			}
		}
		else {
			submitImage(img);
		}
	}

	@Override
	public void removeImage(String img) {
		Log.d(TAG, "Removing image: " + img);
		viewModel.removeImage(img);
	}

	@Override
	public void submitImage(String img) {
		Log.d(TAG, "Adding image: " + img);
		viewModel.addImage(img);
	}

	public void startProcessing(View v) {
		ArrayList<String> imagesToProcess = viewModel.getImagePaths();
		if (imagesToProcess.size() == 0) {
			Log.d(TAG, "No images to process!");
			return;
		}

		Bundle args = new Bundle();
		args.putStringArrayList(IMAGE_DIRS_ARGUMENT_KEY, imagesToProcess);
		Navigation.findNavController(v).navigate(
				R.id.action_galleryFragment_to_confirmImagesFragment,
				null);
	}

}
