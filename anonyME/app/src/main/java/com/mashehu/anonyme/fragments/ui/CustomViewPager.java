package com.mashehu.anonyme.fragments.ui;

import android.content.Context;
import android.util.AttributeSet;
import android.util.Log;
import android.view.MotionEvent;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.lifecycle.LiveData;
import androidx.viewpager.widget.ViewPager;

public class CustomViewPager extends ViewPager {
	private LiveData<Boolean> pagingEnabled;

	public CustomViewPager(@NonNull Context context) {
		super(context);
	}

	public CustomViewPager(@NonNull Context context, @Nullable AttributeSet attrs) {
		super(context, attrs);
	}

	@Override
	public boolean onInterceptTouchEvent(MotionEvent ev) {
		Log.d("CustomViewPager", "paging enabled: "+ pagingEnabled);
		if (isPagingEnabled()) {
			return super.onInterceptTouchEvent(ev);
		}
		return false;
	}

	public boolean isPagingEnabled() {
		return pagingEnabled.getValue();
	}

	public void setPagingEnabled(LiveData<Boolean> pagingEnabled) {
		this.pagingEnabled = pagingEnabled;
	}
}
