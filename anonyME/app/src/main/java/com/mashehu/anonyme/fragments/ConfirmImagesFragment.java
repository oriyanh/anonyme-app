package com.mashehu.anonyme.fragments;


import android.os.Bundle;

import androidx.activity.OnBackPressedCallback;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;
import androidx.lifecycle.ViewModelProviders;
import androidx.navigation.Navigation;
import androidx.recyclerview.widget.DefaultItemAnimator;
import androidx.recyclerview.widget.ItemTouchHelper;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.PagerSnapHelper;
import androidx.recyclerview.widget.RecyclerView;
import androidx.recyclerview.widget.SnapHelper;

import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;

import com.google.android.material.floatingactionbutton.FloatingActionButton;
import com.mashehu.anonyme.R;
import com.mashehu.anonyme.fragments.ui.ConfirmImageLargeAdapter;
import com.mashehu.anonyme.fragments.ui.ImageData;
import com.mashehu.anonyme.fragments.ui.SwipeToDeleteCallback;

import java.util.ArrayList;

import static com.mashehu.anonyme.common.Constants.IMAGE_DIRS_ARGUMENT_KEY;
import static com.mashehu.anonyme.common.Utilities.processImages;


/**
 * A simple {@link Fragment} subclass.
 */
public class ConfirmImagesFragment extends Fragment {
	public static final String TAG = "anonyme.ConfirmImagesFragment";
	FloatingActionButton sendButton;
	RecyclerView recyclerView;
	AppViewModel viewModel;

	public ConfirmImagesFragment() {
		// Required empty public constructor
	}


	@Override
	public View onCreateView(LayoutInflater inflater, ViewGroup container,
							 Bundle savedInstanceState) {
		// Inflate the layout for this fragment
		return inflater.inflate(R.layout.fragment_confirm_images, container, false);
	}

	@Override
	public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
		super.onViewCreated(view, savedInstanceState);
		assert getActivity() != null;

		sendButton = view.findViewById(R.id.fabSendButton);
		viewModel = ViewModelProviders.of(getActivity()).get(AppViewModel.class);
		Log.d(TAG, "tab in previous fragment: " + viewModel.getCurrentTab());
		assert getView() != null;
		assert getActivity() != null;
		assert getArguments() != null;

		ArrayList<String> imagePaths = viewModel.getImagePaths();
//		ArrayList<String> imagePaths = getArguments().getStringArrayList(IMAGE_DIRS_ARGUMENT_KEY);
		ArrayList<ImageData> images = new ArrayList<>();
		for (String img : imagePaths) {
			ImageData imageData = new ImageData();
			imageData.setImagePath(img);
			images.add(imageData);
		}


		setupRecyclerView(images);


		if (getArguments() != null) {
			ArrayList<String> imageDirs = getArguments().getStringArrayList(IMAGE_DIRS_ARGUMENT_KEY);
			if (imageDirs != null)
				Log.d(TAG, String.format("Number of images: %s\nFirst image: %s", imageDirs.size(), imageDirs.get(0)));
		}
		setupListeners();
		requireActivity()
				.getOnBackPressedDispatcher()
				.addCallback(new OnBackPressedCallback(true) {
					@Override
					public void handleOnBackPressed() {
						switch (viewModel.currentTab) {
							case 0:
								Navigation.findNavController(view).navigate(R.id.action_confirmImagesFragment_to_galleryFragment);
								break;
							case 1:
								Navigation.findNavController(view).navigate(R.id.action_confirmImagesFragment2_to_cameraCaptureFragment);
								break;
							default:
								assert getActivity() != null;
								getActivity().finish();

						}
					}
				});
	}

	private void setupListeners() {
		sendButton.setOnClickListener(v -> {
			ConfirmImageLargeAdapter adapter = (ConfirmImageLargeAdapter) recyclerView.getAdapter();
			processImages(getActivity().getApplicationContext(), adapter.getImagePaths());
			getActivity().finish();
		});


	}

	private void setupRecyclerView(ArrayList<ImageData> images) {
		recyclerView = getActivity().findViewById(R.id.confirmImagesRecyclerView);
		ConfirmImageLargeAdapter adapter = new ConfirmImageLargeAdapter(getActivity().getApplicationContext(), images);
		SnapHelper snapHelper = new PagerSnapHelper();
		RecyclerView.LayoutManager layoutManager = new LinearLayoutManager(getContext(), LinearLayoutManager.HORIZONTAL, false);
		recyclerView.setLayoutManager(layoutManager);
		snapHelper.attachToRecyclerView(recyclerView);
		recyclerView.setItemAnimator(new DefaultItemAnimator());
		recyclerView.setAdapter(adapter);
		ItemTouchHelper itemTouchHelper = new ItemTouchHelper(new SwipeToDeleteCallback(adapter));
		itemTouchHelper.attachToRecyclerView(recyclerView);
	}

}


