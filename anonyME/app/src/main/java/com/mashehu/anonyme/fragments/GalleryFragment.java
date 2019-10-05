package com.mashehu.anonyme.fragments;

import android.os.Bundle;

import androidx.activity.OnBackPressedCallback;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;
import androidx.lifecycle.ViewModelProviders;
import androidx.navigation.Navigation;
import androidx.recyclerview.widget.RecyclerView;

import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.TextView;

import com.mashehu.anonyme.R;
import com.mashehu.anonyme.fragments.ui.AutoSpanGridLayoutManager;
import com.mashehu.anonyme.fragments.ui.RecyclerUtils;


import java.util.ArrayList;

import static com.mashehu.anonyme.common.Constants.IMAGE_DIRS_ARGUMENT_KEY;

public class GalleryFragment extends Fragment implements RecyclerUtils.ThumbnailCallback {
	public static final String TAG = "anonyme.GalleryFragment";
	private RecyclerView galleryRecyclerView;
	private ImageView cancelBtn;
	private TextView sendImagesBtn;
	private TextView multipleSelectionBtn;
	private TextView selectAllBtn;
	private TextView unselectAllBtn;
	private AppViewModel viewModel;

	@Override
	public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container,
							 @Nullable Bundle savedInstanceState) {
		return inflater.inflate(R.layout.fragment_gallery, container, false);
	}

	@Override
	public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
		super.onViewCreated(view, savedInstanceState);
		assert getActivity() != null;
		viewModel = ViewModelProviders.of(getActivity()).get(AppViewModel.class);
		viewModel.setPagingEnabled(true);
		sendImagesBtn = view.findViewById(R.id.sendImageBtn);
		cancelBtn = view.findViewById(R.id.cancelSelectionButton);
		multipleSelectionBtn = view.findViewById(R.id.multipleSelectionButton);
		selectAllBtn = view.findViewById(R.id.selectAllBtn);
		unselectAllBtn = view.findViewById(R.id.unselectAllButton);
		viewModel.getMultipleSelectionMode().observe(getActivity(), this::toggleSelectionMode);
		viewModel.getImages().observe(getActivity(), imageData -> {
			if (imageData.size() > 0 && viewModel.isMultipleSelectionMode()) {
				sendImagesBtn.setVisibility(View.VISIBLE);
			}
			else {
				sendImagesBtn.setVisibility(View.INVISIBLE);
			}
		});

		if (viewModel.getImagePaths().size() > 0) {
			sendImagesBtn.setVisibility(View.VISIBLE);
		}
		else {
			sendImagesBtn.setVisibility(View.INVISIBLE);
		}
		sendImagesBtn.setOnClickListener(this::showPreviewFragment);
		cancelBtn.setOnClickListener(v -> viewModel.setMultipleSelectionMode(false));
		multipleSelectionBtn.setOnClickListener(v -> viewModel.setMultipleSelectionMode(true));
		// todo add select/unselect all on click observers

		galleryRecyclerView = getActivity().findViewById(R.id.galleryRecyclerView);
		AutoSpanGridLayoutManager layoutManager = new AutoSpanGridLayoutManager(getActivity().getApplicationContext(), 250);
		galleryRecyclerView.setLayoutManager(layoutManager);

		ArrayList<RecyclerUtils.ImageData> images = viewModel.getGalleryImages();
		RecyclerUtils.ThumbnailAdapter adapter = new RecyclerUtils.ThumbnailAdapter(getActivity().getApplicationContext(), this, images);
		galleryRecyclerView.setAdapter(adapter);
		requireActivity().getOnBackPressedDispatcher().addCallback(new OnBackPressedCallback(true) {
			@Override
			public void handleOnBackPressed() {
				getActivity().finish();
			}
		});
	}

	public void toggleSelectionMode(boolean isMultipleSelection) {
		if (isMultipleSelection) {
			cancelBtn.setVisibility(View.VISIBLE);
			selectAllBtn.setVisibility(View.VISIBLE);
			unselectAllBtn.setVisibility(View.VISIBLE);
			multipleSelectionBtn.setVisibility(View.INVISIBLE);
		}
		else {
			viewModel.clearImages();
			//todo make sure checkmark is not shown anymore
			cancelBtn.setVisibility(View.INVISIBLE);
			selectAllBtn.setVisibility(View.INVISIBLE);
			unselectAllBtn.setVisibility(View.INVISIBLE);
			multipleSelectionBtn.setVisibility(View.VISIBLE);
		}
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

	public void showPreviewFragment(View v) {
		ArrayList<String> imagesToProcess = viewModel.getImagePaths();
		if (imagesToProcess.size() == 0) {
			Log.d(TAG, "No images to process!");
			return;
		}

		Bundle args = new Bundle();
		args.putStringArrayList(IMAGE_DIRS_ARGUMENT_KEY, imagesToProcess);
		Navigation.findNavController(getView()).navigate(
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
