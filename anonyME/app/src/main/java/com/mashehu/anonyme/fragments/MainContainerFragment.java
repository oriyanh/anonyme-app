package com.mashehu.anonyme.fragments;


import android.content.SharedPreferences;
import android.os.Bundle;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;
import androidx.lifecycle.ViewModelProviders;
import androidx.navigation.Navigation;
import androidx.viewpager.widget.ViewPager;

import android.preference.PreferenceManager;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;

import com.ToxicBakery.viewpager.transforms.CubeOutTransformer;
import com.mashehu.anonyme.R;
import com.mashehu.anonyme.fragments.ui.CustomViewPager;
import com.mashehu.anonyme.fragments.ui.RecyclerUtils;

import static androidx.fragment.app.FragmentStatePagerAdapter.BEHAVIOR_RESUME_ONLY_CURRENT_FRAGMENT;
import static com.mashehu.anonyme.common.Constants.SP_IS_PROCESSING_KEY;

/**
 * A simple {@link Fragment} subclass.
 */
public class MainContainerFragment extends Fragment {

	AppViewModel viewModel;
	CustomViewPager fragmentViewPager;
	RecyclerUtils.FragmentPagerAdapter adapter;

	public MainContainerFragment() {
		// Required empty public constructor
	}


	@Override
	public View onCreateView(LayoutInflater inflater, ViewGroup container,
							 Bundle savedInstanceState) {
		// Inflate the layout for this fragment
		return inflater.inflate(R.layout.fragment_main_container, container, false);
	}

	@Override
	public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
		super.onViewCreated(view, savedInstanceState);

		SharedPreferences sp = PreferenceManager.getDefaultSharedPreferences(getContext().getApplicationContext());
		sp.edit().putBoolean(SP_IS_PROCESSING_KEY, false).commit();

		navigateIfNecessary(sp, view);

		sp.registerOnSharedPreferenceChangeListener((sharedPreferences, key) -> {
			if (key.equals(SP_IS_PROCESSING_KEY)) {
				navigateIfNecessary(sharedPreferences, view);
			}
		});

		assert getActivity() != null;
		viewModel = ViewModelProviders.of(getActivity()).get(AppViewModel.class);
		viewModel.setCurrentTab(1); // Makes sure app will start on camera capture mode

		fragmentViewPager = view.findViewById(R.id.fragmentViewPager);
//        adapter = new ViewPagerFragmentAdapter(getSupportFragmentManager(), getLifecycle(), fragments);
		adapter = new RecyclerUtils.FragmentPagerAdapter(getChildFragmentManager(),
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
		assert getActivity() != null;
		fragmentViewPager.setPageTransformer(true, new CubeOutTransformer());
		fragmentViewPager.setPagingEnabled(viewModel.getPagingEnabled());
	}

	public void navigateIfNecessary(SharedPreferences sp, View v) {
		boolean isProcessing = sp.getBoolean(SP_IS_PROCESSING_KEY, false);
		if (isProcessing) {
			Navigation.findNavController(v).navigate(R.id.action_mainContainerFragment2_to_loadingScreenFragment);
		}
	}
}
