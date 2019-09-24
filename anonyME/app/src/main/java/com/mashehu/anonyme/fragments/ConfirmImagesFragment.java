package com.mashehu.anonyme.fragments;


import android.os.Bundle;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;
import androidx.navigation.Navigation;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageButton;
import android.widget.TextView;

import com.mashehu.anonyme.R;

import java.util.ArrayList;

import static com.mashehu.anonyme.common.Constants.IMAGE_DIRS_ARGUMENT_KEY;

import java.util.ArrayList;

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
            ArrayList<String> imageDirs = getArguments().getStringArrayList(IMAGE_DIRS_ARGUMENT_KEY);
            if (imageDirs != null)
                filesDirTextView.setText(String.format("%s%s", imageDirs.size(), imageDirs.get(0)));
        }

        ImageButton returnButton = getView().findViewById(R.id.return_button);
        returnButton.setOnClickListener((v) -> {
                Bundle args = new Bundle();
                args.putBoolean("isBulkCapture", true);
                Navigation.findNavController(view).navigate(
                        R.id.action_confirmImagesFragment_to_cameraCaptureFragment, args);
            });
    }
}
