package com.mashehu.anonyme.fragments.ui;

import androidx.annotation.NonNull;
import androidx.fragment.app.Fragment;
import androidx.fragment.app.FragmentManager;
import androidx.fragment.app.FragmentStatePagerAdapter;

import com.mashehu.anonyme.fragments.CameraCaptureFragment;
import com.mashehu.anonyme.fragments.GalleryFragment;

import java.util.ArrayList;

public class ViewPagerFragmentAdapter extends FragmentStatePagerAdapter {

	public ArrayList<Fragment> fragments;

	public ViewPagerFragmentAdapter(@NonNull FragmentManager fm, int behavior) {
		super(fm, behavior);

	}

	@NonNull
	@Override
	public Fragment getItem(int position) {
		if (position == 1) {
			return new CameraCaptureFragment();
		}
		return new GalleryFragment();
	}

	@Override
	public int getCount() {
		return 2;
	}
}
