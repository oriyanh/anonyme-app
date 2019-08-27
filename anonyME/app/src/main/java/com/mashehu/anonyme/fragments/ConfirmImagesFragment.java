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
public class ConfirmImagesFragment extends Fragment {


    public ConfirmImagesFragment() {
        // Required empty public constructor
    }


    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                             Bundle savedInstanceState) {
        // Inflate the layout for this fragment
        return inflater.inflate(R.layout.fragment_confirm_images, container, false);
    }

}
