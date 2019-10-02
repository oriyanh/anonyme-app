package com.mashehu.anonyme.fragments;


import android.os.Bundle;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;
import androidx.lifecycle.ViewModelProviders;
import androidx.viewpager.widget.ViewPager;

import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;

import com.mashehu.anonyme.R;
import com.mashehu.anonyme.fragments.ui.ViewPagerFragmentAdapter;

import java.util.ArrayList;

import static androidx.fragment.app.FragmentStatePagerAdapter.BEHAVIOR_RESUME_ONLY_CURRENT_FRAGMENT;

/**
 * A simple {@link Fragment} subclass.
 */
public class ContainerFragment extends Fragment {
	ViewPager fragmentViewPager;
	ArrayList<Fragment> fragments;
	ViewPagerFragmentAdapter adapter;
	AppViewModel viewModel;
	public ContainerFragment() {
		// Required empty public constructor
	}


	@Override
	public View onCreateView(LayoutInflater inflater, ViewGroup container,
							 Bundle savedInstanceState) {
		// Inflate the layout for this fragment
		return inflater.inflate(R.layout.fragment_container, container, false);
	}

	@Override
	public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
		super.onViewCreated(view, savedInstanceState);
		assert getActivity() != null;

		viewModel = ViewModelProviders.of(getActivity()).get(AppViewModel.class);

		fragments = new ArrayList<>();
		fragments.add(new CameraCaptureFragment());
		fragments.add(new GalleryFragment());
		fragmentViewPager = view.findViewById(R.id.fragmentViewPager);
//        adapter = new ViewPagerFragmentAdapter(getSupportFragmentManager(), getLifecycle(), fragments);
		adapter = new ViewPagerFragmentAdapter(requireActivity().getSupportFragmentManager(),
				BEHAVIOR_RESUME_ONLY_CURRENT_FRAGMENT);
//        fragmentViewPager.setOrientation(ViewPager2.ORIENTATION_HORIZONTAL);
		fragmentViewPager.setAdapter(adapter);
		Log.d("anonyme.ContainerFragment", "Previous tab position: " + viewModel.getCurrentTab());
		if (viewModel.getCurrentTab() != -1) {
			fragmentViewPager.setCurrentItem(viewModel.getCurrentTab());
		}
		else {
			fragmentViewPager.setCurrentItem(1);
			viewModel.setCurrentTab(1);
		}

		fragmentViewPager.addOnPageChangeListener(new ViewPager.OnPageChangeListener() {
			@Override
			public void onPageScrolled(int position, float positionOffset, int positionOffsetPixels) {

			}

			@Override
			public void onPageSelected(int position) {
				Log.d("anonyme.ContainerFragment", "Current tab position: " + position);
				viewModel.setCurrentTab(position);
			}

			@Override
			public void onPageScrollStateChanged(int state) {

			}
		});
	}
}
