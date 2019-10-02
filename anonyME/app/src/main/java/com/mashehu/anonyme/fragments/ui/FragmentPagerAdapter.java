package com.mashehu.anonyme.fragments.ui;

import androidx.annotation.NonNull;
import androidx.fragment.app.Fragment;
import androidx.fragment.app.FragmentManager;
import androidx.fragment.app.FragmentStatePagerAdapter;

import com.mashehu.anonyme.fragments.CameraCaptureContainerFragment;
import com.mashehu.anonyme.fragments.GalleryContainerFragment;

import java.util.ArrayList;

public class FragmentPagerAdapter extends FragmentStatePagerAdapter {

	public ArrayList<Fragment> fragments;

	public FragmentPagerAdapter(@NonNull FragmentManager fm, int behavior) {
		super(fm, behavior);

	}

	@NonNull
	@Override
	public Fragment getItem(int position) {
		if (position == 1) {
			return new CameraCaptureContainerFragment();
		}
		return new GalleryContainerFragment();
	}

	@Override
	public int getCount() {
		return 2;
	}
}
