package com.mashehu.anonyme.fragments;


import android.os.Bundle;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;
import androidx.navigation.Navigation;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;

import com.mashehu.anonyme.R;

import static com.mashehu.anonyme.common.Utilities.isProcessing;

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

//		SharedPreferences sp = PreferenceManager.getDefaultSharedPreferences(getContext().getApplicationContext());
		navigateIfNecessary(view);

//		sp.registerOnSharedPreferenceChangeListener((sharedPreferences, key) -> {
//			if (key.equals(SP_IS_PROCESSING_KEY)) {
//				navigateIfNecessary(sharedPreferences, view);
//			}
//		});
	}

	public void navigateIfNecessary(View v) {
//		boolean isProcessing = sp.getBoolean(SP_IS_PROCESSING_KEY, false);
		if (!isProcessing(getContext().getApplicationContext())) {
			Navigation.findNavController(v).navigate(
					R.id.action_loadingScreenFragment_to_mainContainerFragment2);
		}
	}

	@Override
	public void onPause() {
		super.onPause();
		getActivity().finish();
	}
}

