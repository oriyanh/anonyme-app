package com.mashehu.anonyme.fragments.ui;

import androidx.annotation.NonNull;

import com.bumptech.glide.annotation.GlideExtension;
import com.bumptech.glide.annotation.GlideOption;
import com.bumptech.glide.request.BaseRequestOptions;

@GlideExtension
public class AnonyMEGlideExtension {
	private AnonyMEGlideExtension() {}

	@GlideOption
	@NonNull
	public static BaseRequestOptions<?> galleryThumbnail(BaseRequestOptions<?> options)
	{
		return options
				.fitCenter()
				.centerCrop();
//				.override(50, 50);
	}
}