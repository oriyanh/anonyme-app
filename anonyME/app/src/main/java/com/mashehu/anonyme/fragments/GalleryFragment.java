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
import com.mashehu.anonyme.fragments.ui.RecyclerUtils;

import java.util.ArrayList;

import static com.mashehu.anonyme.common.Constants.IMAGE_DIRS_ARGUMENT_KEY;

public class GalleryFragment extends Fragment implements RecyclerUtils.ThumbnailCallback {
	public static final String TAG = "anonyme.GalleryFragment";
	private RecyclerView galleryRecyclerView;
	private Button sendImagesBtn;
	private AppViewModel viewModel;

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
		viewModel.setPagingEnabled(true);
		sendImagesBtn = getActivity().findViewById(R.id.sendImageBtn);

		sendImagesBtn.setOnClickListener(this::startProcessing);

		galleryRecyclerView = getActivity().findViewById(R.id.galleryRecyclerView);
		GridLayoutManager layoutManager = new GridLayoutManager(getActivity().getApplicationContext(), 4);
		galleryRecyclerView.setLayoutManager(layoutManager);

		ArrayList<RecyclerUtils.ImageData> images = Utilities.getGalleryContent();
		RecyclerUtils.ThumbnailAdapter adapter = new RecyclerUtils.ThumbnailAdapter(getActivity().getApplicationContext(), this, images);
		galleryRecyclerView.setAdapter(adapter);
		requireActivity().getOnBackPressedDispatcher().addCallback(new OnBackPressedCallback(true) {
			@Override
			public void handleOnBackPressed() {
				getActivity().finish();
			}
		});
	}

	@Override
	public void removeImage(RecyclerUtils.ImageData img) {
		Log.d(TAG, "Removing image: " + img.getImagePath());
		viewModel.removeImage(img.getImagePath());
	}


	@Override
	public void addImage(RecyclerUtils.ImageData img) {
		Log.d(TAG, "Adding image: " + img.getImagePath());
		viewModel.addImage(img.getImagePath());
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

	@Override
	public boolean isMultipleSelection() {
		return viewModel.isMultipleSelectionMode();
	}

	@Override
	public void setMultipleSelection(boolean isMultipleSelection) {
		viewModel.setMultipleSelectionMode(isMultipleSelection);
	}

}
