package com.mashehu.anonyme.fragments;


import android.os.Bundle;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

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

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);

        assert getView() != null;
        TextView filesDirTextView = getView().findViewById(R.id.files_dir_text_view);
        if (getArguments() != null)
        {
            String[] imageDirs = getArguments().getStringArray("imageDirs");
            filesDirTextView.setText(imageDirs.length + imageDirs[0]);
        }
    }
}
