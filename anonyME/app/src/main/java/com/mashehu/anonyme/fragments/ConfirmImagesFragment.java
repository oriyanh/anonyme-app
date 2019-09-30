package com.mashehu.anonyme.fragments;


import android.os.Bundle;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.core.view.ViewCompat;
import androidx.fragment.app.Fragment;
import androidx.navigation.Navigation;
import androidx.recyclerview.widget.ItemTouchHelper;
import androidx.viewpager2.widget.ViewPager2;

import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageButton;

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
	ViewPager2 viewPager;
	FloatingActionButton sendButton;
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

		viewPager = getActivity().findViewById(R.id.confirmImagesviewPager);
		ArrayList<String> imagePaths = getArguments().getStringArrayList(IMAGE_DIRS_ARGUMENT_KEY);
		ArrayList<ImageData> images = new ArrayList<>();
		for (String img : imagePaths) {
			ImageData imageData = new ImageData();
			imageData.setImagePath(img);
			images.add(imageData);
		}
		ConfirmImageLargeAdapter adapter = new ConfirmImageLargeAdapter(getActivity().getApplicationContext(), images);
		viewPager.setOrientation(ViewPager2.ORIENTATION_HORIZONTAL);
		viewPager.setClipToPadding(false);
		viewPager.setClipChildren(false);
		viewPager.setOffscreenPageLimit(3);

		ItemTouchHelper itemTouchHelper = new ItemTouchHelper(new SwipeToDeleteCallback(adapter));
		viewPager.addItemDecoration(itemTouchHelper);
		viewPager.setAdapter(adapter);

		sendButton = view.findViewById(R.id.fabSendButton);
		sendButton.setOnClickListener(v -> {
			processImages(getActivity().getApplicationContext(), adapter.getImagePaths());
		});

		int offsetPx = getResources().getDimensionPixelOffset(R.dimen.offset);
		int pageMarginPx = getResources().getDimensionPixelOffset(R.dimen.pageMargin);
		viewPager.setPageTransformer((page, position) -> {
			ViewPager2 vp = (ViewPager2) page.getParent().getParent();
			float offset = position * -(2 * offsetPx + pageMarginPx);
			if (vp.getOrientation() == ViewPager2.ORIENTATION_HORIZONTAL) {
				if (ViewCompat.getLayoutDirection(vp) == ViewCompat.LAYOUT_DIRECTION_RTL) {
					page.setTranslationX(-offset);
				}
				else {
					page.setTranslationX(offset);
				}
			}
			else {
				page.setTranslationY(offset);
			}
		});

		if (getArguments() != null) {
			ArrayList<String> imageDirs = getArguments().getStringArrayList(IMAGE_DIRS_ARGUMENT_KEY);
			if (imageDirs != null)
				Log.d(TAG, String.format("Number of images: %s\nFirst image: %s", imageDirs.size(), imageDirs.get(0)));
		}
		setupListeners(view);

	}

	private void setupListeners(@NonNull View view) {
		ImageButton returnButton = getView().findViewById(R.id.return_button);
		returnButton.setOnClickListener((v) -> {
			Bundle args = new Bundle();
			args.putBoolean("isBulkCapture", true);
			Navigation.findNavController(view).navigate(
					R.id.action_confirmImagesFragment_to_cameraCaptureFragment, args);
		});


	}
}
