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
public class ShareContainerFragment extends Fragment {


    public ShareContainerFragment() {
        // Required empty public constructor
    }


    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                             Bundle savedInstanceState) {
        // Inflate the layout for this fragment
        return inflater.inflate(R.layout.fragment_share_container, container, false);
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);

        int navActionId;
        if (isProcessing(getContext().getApplicationContext()))
        {
            navActionId = R.id.action_shareContainerFragment_to_loadingScreenFragment2;
        }
        else
        {
            navActionId = R.id.action_shareContainerFragment_to_confirmImagesFragment3;
        }
        Navigation.findNavController(view).navigate(navActionId);
    }
}
