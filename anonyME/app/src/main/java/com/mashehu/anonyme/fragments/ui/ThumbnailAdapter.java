package com.mashehu.anonyme.fragments.ui;

import android.content.Context;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

import com.mashehu.anonyme.R;
import com.mashehu.anonyme.fragments.GallerySelectionHandler;

import java.util.ArrayList;

public class ThumbnailAdapter extends RecyclerView.Adapter<ThumbnailViewHolder> {
	private static final String TAG = "anonyme.ThumbnailAdapter";
	private ArrayList<ImageData> imageList;
	private Context context;
	private GallerySelectionHandler selector;

	public ThumbnailAdapter(Context context, GallerySelectionHandler selector, ArrayList<ImageData> images) {
		this.imageList = images;
		this.context = context;
		this.selector = selector;
	}

	@NonNull
	@Override
	public ThumbnailViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
		View view = LayoutInflater.from(parent.getContext()).inflate(R.layout.item_image_thumbnail, parent, false);
		return new ThumbnailViewHolder(view);
	}

	@Override
	public void onBindViewHolder(@NonNull ThumbnailViewHolder holder, int position) {
		String img = imageList.get(position).getImagePath();

		holder.img.setOnClickListener(v -> {
			Log.d(TAG, "Normal click on image");
			if (!selector.getSelectionMode()) {
				selector.toggleCheckbox(null, img);
				selector.startProcessing(v);
			}
			else {
				selector.toggleCheckbox(holder, img);
			}
		});

		holder.img.setOnLongClickListener(v -> {
			Log.d(TAG, "Long click on image");

			if (!selector.getSelectionMode()) {
				selector.setSelectionMode(true);
			}
			selector.toggleCheckbox(holder, img);
			return true;
		});

		GlideApp.with(context)
				.load(img)
				.galleryThumbnail()
				.into(holder.img);
	}

	@Override
	public int getItemCount() {
		return imageList.size();
	}

}
