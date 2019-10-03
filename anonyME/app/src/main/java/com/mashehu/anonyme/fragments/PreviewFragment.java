package com.mashehu.anonyme.fragments;


import android.os.Bundle;

import androidx.activity.OnBackPressedCallback;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;
import androidx.lifecycle.LiveData;
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
import com.mashehu.anonyme.fragments.ui.RecyclerUtils;

import java.util.ArrayList;

import static com.mashehu.anonyme.common.Utilities.processImages;


/**
 * A simple {@link Fragment} subclass.
 */
public class PreviewFragment extends Fragment implements RecyclerUtils.PreviewItemsCallback {
	public static final String TAG = "anonyme.PreviewFragment";
	private FloatingActionButton sendButton;
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

		sendButton = view.findViewById(R.id.fabSendButton);
		viewModel = ViewModelProviders.of(getActivity()).get(AppViewModel.class);
		viewModel.setPagingEnabled(false);
		Log.d(TAG, "tab in previous fragment: " + viewModel.getCurrentTab());
		assert getView() != null;
		assert getActivity() != null;
		assert getArguments() != null;

		LiveData<ArrayList<RecyclerUtils.ImageData>> images = viewModel.getImages();

		images.observe(this, imageData -> {
			if (imageData.size() == 0) {
				requireActivity().getOnBackPressedDispatcher().onBackPressed();
			}
		});

		setupRecyclerView(images.getValue());
		setupListeners(view);

	}

	private void setupListeners(View view) {
		sendButton.setOnClickListener(v -> {
			processImages(getActivity().getApplicationContext(), viewModel.getImagePaths());
			getActivity().finish();
		});

		requireActivity()
				.getOnBackPressedDispatcher()
				.addCallback(new OnBackPressedCallback(true) {
					@Override
					public void handleOnBackPressed() {
						switch (viewModel.getCurrentTab()) {
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
	}
}

