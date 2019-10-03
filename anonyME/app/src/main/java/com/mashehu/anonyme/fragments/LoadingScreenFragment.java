package com.mashehu.anonyme.fragments;


import android.content.SharedPreferences;
import android.os.Bundle;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;
import androidx.navigation.Navigation;

import android.preference.PreferenceManager;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;

import com.mashehu.anonyme.R;

import static com.mashehu.anonyme.common.Constants.SP_IS_PROCESSING_KEY;

/**
 * A simple {@link Fragment} subclass.
 */
public class LoadingScreenFragment extends Fragment {


	public LoadingScreenFragment() {
		// Required empty public constructor
	}


	@Override
	public View onCreateView(LayoutInflater inflater, ViewGroup container,
							 Bundle savedInstanceState) {
		// Inflate the layout for this fragment
		return inflater.inflate(R.layout.fragment_loading_screen, container, false);
	}

	@Override
	public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
		super.onViewCreated(view, savedInstanceState);

		SharedPreferences sp = PreferenceManager.getDefaultSharedPreferences(getContext());
		navigateIfNecessary(sp, view);

		sp.registerOnSharedPreferenceChangeListener((sharedPreferences, key) -> {
			if (key.equals(SP_IS_PROCESSING_KEY)) {
				navigateIfNecessary(sharedPreferences, view);
			}
		});
	}

	public void navigateIfNecessary(SharedPreferences sp, View v) {
		boolean isProcessing = sp.getBoolean(SP_IS_PROCESSING_KEY, false);
		if (!isProcessing) {
			Navigation.findNavController(v).navigate(R.id.action_loadingScreenFragment_to_mainContainerFragment2);
		}
	}
}

