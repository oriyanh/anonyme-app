package com.mashehu.anonyme.fragments;

import android.os.Bundle;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;
import androidx.lifecycle.Observer;
import androidx.lifecycle.ViewModelProviders;
import androidx.navigation.Navigation;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;

import com.mashehu.anonyme.R;

/**
 * A simple {@link Fragment} subclass.
 */
public class SplashScreenFragment extends Fragment {

	AppViewModel viewModel;
	public SplashScreenFragment() {
		// Required empty public constructor
	}


	@Override
	public View onCreateView(LayoutInflater inflater, ViewGroup container,
							 Bundle savedInstanceState) {
		// Inflate the layout for this fragment
		return inflater.inflate(R.layout.fragment_splash_screen, container, false);
	}

	@Override
	public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
		super.onViewCreated(view, savedInstanceState);
		viewModel = ViewModelProviders.of(requireActivity()).get(AppViewModel.class);
		viewModel.galleryReady().observe(requireActivity(), new Observer<Boolean>() {
			@Override
			public void onChanged(Boolean aBoolean) {
				if (aBoolean) {
					Navigation.findNavController(view).navigate(R.id.action_splashScreenFragment_to_mainContainerFragment2);
				}
			}
		});
		viewModel.loadGallery();
//		viewModel.loadAssets(requireActivity().getApplicationContext());
	}

}
