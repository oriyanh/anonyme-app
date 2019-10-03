package com.mashehu.anonyme.fragments.ui;

import android.content.Context;
import android.util.AttributeSet;
import android.view.MotionEvent;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.viewpager.widget.ViewPager;

public class CustomViewPager extends ViewPager {
	private boolean pagingEnabled;

	public CustomViewPager(@NonNull Context context) {
		super(context);
	}

	public CustomViewPager(@NonNull Context context, @Nullable AttributeSet attrs) {
		super(context, attrs);
	}

	@Override
	public boolean onInterceptTouchEvent(MotionEvent ev) {
		if (isPagingEnabled()) {
			return super.onInterceptTouchEvent(ev);
		}
		return false;
	}

	@Override
	public boolean onTouchEvent(MotionEvent ev) {
		if (isPagingEnabled()) {
			return super.onTouchEvent(ev);
		}
		return false;
	}

	public boolean isPagingEnabled() {
		return pagingEnabled;
	}

	public void setPagingEnabled(boolean pagingEnabled) {
		this.pagingEnabled = pagingEnabled;
	}
}
