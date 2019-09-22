package com.mashehu.anonyme.fragments;

import androidx.lifecycle.ViewModelProviders;

import android.os.Bundle;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;
import androidx.recyclerview.widget.GridLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;

import com.mashehu.anonyme.R;
import com.mashehu.anonyme.common.Utilities;
import com.mashehu.anonyme.fragments.ui.ImageData;
import com.mashehu.anonyme.fragments.ui.ThumbnailAdapter;

import java.util.ArrayList;

public class GalleryFragment extends Fragment {

    private GalleryViewModel mViewModel;
    private ImageView thumbnail;
    RecyclerView recyclerView;
    GridLayoutManager layoutManager;
    public static GalleryFragment newInstance() {
        return new GalleryFragment();
    }

    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container,
                             @Nullable Bundle savedInstanceState) {
        return inflater.inflate(R.layout.gallery_fragment, container, false);
    }

    @Override
    public void onActivityCreated(@Nullable Bundle savedInstanceState) {
        super.onActivityCreated(savedInstanceState);
        mViewModel = ViewModelProviders.of(this).get(GalleryViewModel.class);
        // TODO: Use the ViewModel

        assert getActivity() != null;

        thumbnail = getActivity().findViewById(R.id.thumbnail_view);
        recyclerView = getActivity().findViewById(R.id.galleryRecyclerView);
        layoutManager = new GridLayoutManager(getActivity().getApplicationContext(), 4);
        recyclerView.setLayoutManager(layoutManager);

        ArrayList<ImageData> images = Utilities.getGalleryContent();
        ThumbnailAdapter adapter = new ThumbnailAdapter(getActivity().getApplicationContext(), images);
        recyclerView.setAdapter(adapter);
    }

}
