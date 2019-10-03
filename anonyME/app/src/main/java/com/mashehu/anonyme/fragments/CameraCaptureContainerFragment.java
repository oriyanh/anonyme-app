package com.mashehu.anonyme.fragments;


import android.os.Bundle;

import androidx.fragment.app.Fragment;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import com.mashehu.anonyme.R;

/**
 * A simple {@link Fragment} subclass.
 */
public class CameraCaptureContainerFragment extends Fragment {


	public CameraCaptureContainerFragment() {
		// Required empty public constructor
	}


	@Override
	public View onCreateView(LayoutInflater inflater, ViewGroup container,
							 Bundle savedInstanceState) {
		return inflater.inflate(R.layout.fragment_camera_capture_container, container, false);

	}

}
