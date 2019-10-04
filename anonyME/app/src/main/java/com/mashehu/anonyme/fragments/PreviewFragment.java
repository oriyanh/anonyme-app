package com.mashehu.anonyme.fragments;


import android.os.Bundle;

import androidx.activity.OnBackPressedCallback;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;
import androidx.lifecycle.LiveData;
import androidx.lifecycle.ViewModelProviders;
import androidx.navigation.NavController;
import androidx.navigation.Navigation;
import androidx.recyclerview.widget.DefaultItemAnimator;
import androidx.recyclerview.widget.ItemTouchHelper;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.PagerSnapHelper;
import androidx.recyclerview.widget.RecyclerView;
import androidx.recyclerview.widget.SnapHelper;

import android.util.Log;
import android.view.Gravity;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.Toast;

import com.mashehu.anonyme.R;
import com.mashehu.anonyme.fragments.ui.RecyclerUtils;

import java.util.ArrayList;

import static com.mashehu.anonyme.common.Utilities.processImages;


/**
 * A simple {@link Fragment} subclass.
 */
public class PreviewFragment extends Fragment implements RecyclerUtils.PreviewItemsCallback {
	public static final String TAG = "anonyme.PreviewFragment";
	private ImageView sendButton;
	private ImageView cancelButton;
	private ImageView addButton;
	private RecyclerView recyclerView;
	private AppViewModel viewModel;
	RecyclerUtils.PreviewImagesAdapter adapter;

	public PreviewFragment() {
		// Required empty public constructor
	}


	@Override
	public View onCreateView(LayoutInflater inflater, ViewGroup container,
							 Bundle savedInstanceState) {
		// Inflate the layout for this fragment
		return inflater.inflate(R.layout.fragment_preview, container, false);
	}

	@Override
	public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
		super.onViewCreated(view, savedInstanceState);
		assert getActivity() != null;

		sendButton = view.findViewById(R.id.sendButton);
		cancelButton = view.findViewById(R.id.cancelButton);
		addButton = view.findViewById(R.id.addMorePhotosButton);


		viewModel = ViewModelProviders.of(getActivity()).get(AppViewModel.class);
		viewModel.setPagingEnabled(false);
		Log.d(TAG, "tab in previous fragment: " + viewModel.getCurrentTab());
		assert getView() != null;
		assert getActivity() != null;
		assert getArguments() != null;

		LiveData<ArrayList<RecyclerUtils.ImageData>> images = viewModel.getImages();

		images.observe(getActivity(), imageData -> {
			if (imageData.size() == 0) {
				viewModel.setBulkCaptureMode(false);
				viewModel.setMultipleSelectionMode(false);
				navigateBack();
			}
		});

		cancelButton.setOnClickListener(v -> viewModel.clearImages());
		setupRecyclerView(images.getValue());
		setupListeners();

	}

	private void setupListeners() {
		sendButton.setOnClickListener(v -> {
			processImages(getActivity().getApplicationContext(), viewModel.getImagePaths());
			getActivity().finish();
		});

		cancelButton.setOnClickListener(v -> {
			viewModel.setBulkCaptureMode(false);
			viewModel.setMultipleSelectionMode(false);
			viewModel.clearImages();
			navigateBack();
		});

		addButton.setOnClickListener(v -> {
			viewModel.setBulkCaptureMode(true);
			viewModel.setMultipleSelectionMode(true);
			navigateBack();
		});

		requireActivity()
				.getOnBackPressedDispatcher()
				.addCallback(getViewLifecycleOwner(), new OnBackPressedCallback(true) {
					@Override
					public void handleOnBackPressed() {
						if (viewModel.getCurrentTab() == -1) {
							requireActivity().finish();
						}
					}
				});


	}

	public void navigateBack() {
		NavController navController;
		if (viewModel.getCurrentTab() == 0) {
			navController = Navigation.findNavController(requireActivity(), R.id.navHostGalleryFragment);
			if (navController.getCurrentDestination().getId() == R.id.confirmImagesFragment) {
				navController.navigate(R.id.action_confirmImagesFragment_to_galleryFragment);
			}
		}
		else if (viewModel.getCurrentTab() == 1) {
			navController =Navigation.findNavController(requireActivity(), R.id.navHostCameraContainer);
			if (navController.getCurrentDestination().getId() == R.id.confirmImagesFragment) {
				navController.navigate(R.id.action_confirmImagesFragment2_to_cameraCaptureFragment);
			}
		}
		else {
			requireActivity().finish();
		}
	}

	private void setupRecyclerView(ArrayList<RecyclerUtils.ImageData> images) {
		recyclerView = getActivity().findViewById(R.id.confirmImagesRecyclerView);
		adapter = new RecyclerUtils.PreviewImagesAdapter(getActivity().getApplicationContext(), images);
		adapter.callback = this;
		RecyclerView.LayoutManager layoutManager = new LinearLayoutManager(getContext(), LinearLayoutManager.HORIZONTAL, false);
		recyclerView.setLayoutManager(layoutManager);
		SnapHelper snapHelper = new PagerSnapHelper();
		snapHelper.attachToRecyclerView(recyclerView);
		recyclerView.setItemAnimator(new DefaultItemAnimator());
		recyclerView.setAdapter(adapter);
		ItemTouchHelper itemTouchHelper = new ItemTouchHelper(new RecyclerUtils.SwipeToDeleteCallback(adapter));
		itemTouchHelper.attachToRecyclerView(recyclerView);
	}

	@Override
	public void removeItem(RecyclerUtils.ImageData img) {
		viewModel.removeImage(img.getImagePath());
		Toast deletedToast = Toast.makeText(getContext(), "REMOVED", Toast.LENGTH_SHORT);
		deletedToast.setGravity(Gravity.BOTTOM, 0, 10);
		deletedToast.show();
	}
}

