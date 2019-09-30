package com.mashehu.anonyme.fragments;


import android.os.Bundle;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;
import androidx.navigation.NavController;
import androidx.navigation.NavDestination;
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
public class ConfirmImagesFragment extends Fragment{
	public static final String TAG = "anonyme.ConfirmImagesFragment";
	//	ViewPager2 viewPager;
	FloatingActionButton sendButton;
	RecyclerView recyclerView;

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

		assert getView() != null;
		assert getActivity() != null;
		assert getArguments() != null;

		ArrayList<String> imagePaths = getArguments().getStringArrayList(IMAGE_DIRS_ARGUMENT_KEY);
		ArrayList<ImageData> images = new ArrayList<>();
		for (String img : imagePaths) {
			ImageData imageData = new ImageData();
			imageData.setImagePath(img);
			images.add(imageData);
//			images.add(imageData);
//			images.add(imageData);
		}


		setupRecyclerView(images);
		sendButton = view.findViewById(R.id.fabSendButton);


		if (getArguments() != null) {
			ArrayList<String> imageDirs = getArguments().getStringArrayList(IMAGE_DIRS_ARGUMENT_KEY);
			if (imageDirs != null)
				Log.d(TAG, String.format("Number of images: %s\nFirst image: %s", imageDirs.size(), imageDirs.get(0)));
		}
		setupListeners(view);

	}

	private void setupListeners(@NonNull View view) {
//		ImageButton returnButton = getView().findViewById(R.id.return_button);
//		returnButton.setOnClickListener((v) -> {
//			Bundle args = new Bundle();
//			args.putBoolean("isBulkCapture", true);
//			NavController navController = Navigation.findNavController(view);
//
//			Navigation.findNavController(view).navigate(
//					R.id.action_confirmImagesFragment_to_cameraCaptureFragment, args);
//		});

		sendButton.setOnClickListener(v -> {
			ConfirmImageLargeAdapter adapter = (ConfirmImageLargeAdapter) recyclerView.getAdapter();
			processImages(getActivity().getApplicationContext(), adapter.getImagePaths());
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
//		viewPager.setOrientation(ViewPager2.ORIENTATION_HORIZONTAL);
//		viewPager.setClipToPadding(false);
//		viewPager.setClipChildren(false);
//		viewPager.setOffscreenPageLimit(3);

		ItemTouchHelper itemTouchHelper = new ItemTouchHelper(new SwipeToDeleteCallback(adapter));
//		viewPager.addItemDecoration(itemTouchHelper);
//		viewPager.setAdapter(adapter);
		itemTouchHelper.attachToRecyclerView(recyclerView);
	}

	private void setupBackButton(@NonNull View view) {
		NavController navController = Navigation.findNavController(view);
		navController.addOnDestinationChangedListener(new NavController.OnDestinationChangedListener() {
			@Override
			public void onDestinationChanged(@NonNull NavController controller, @NonNull NavDestination destination, @Nullable Bundle arguments) {
//				if (destination.getId() == R.id.fragment)
			}
		});
	}

}


