package com.mashehu.anonyme.fragments.ui;

import android.content.Context;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;

import androidx.annotation.NonNull;
import androidx.navigation.Navigation;
import androidx.recyclerview.widget.RecyclerView;

import com.bumptech.glide.Glide;
import com.mashehu.anonyme.R;

import java.util.ArrayList;

public class ThumbnailAdapter extends RecyclerView.Adapter<ThumbnailViewHolder> {
	private ArrayList<ImageData> images;
	private Context context;
	private boolean multipleSelectionMode = false;

	public ThumbnailAdapter(Context context, ArrayList<ImageData> images) {
		this.images = images;
		this.context = context;
	}

	@NonNull
	@Override
	public ThumbnailViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
		View view = LayoutInflater.from(parent.getContext()).inflate(R.layout.item_image_thumbnail, parent, false);
		return new ThumbnailViewHolder(view);
	}

	@Override
	public void onBindViewHolder(@NonNull ThumbnailViewHolder holder, int position) {

		holder.img.setOnClickListener(v -> {
			if (!multipleSelectionMode) {
				Bundle args = new Bundle();
				String img = images.get(position).getImagePath();
				ArrayList<String> images = new ArrayList<>();
				images.add(img);
				args.putStringArrayList("imageDirs", images);
				Navigation.findNavController(v).navigate(
						R.id.action_galleryFragment_to_confirmImagesFragment,
						args);
			}
			else {} //TODO work on multiple selection mode
		});
		holder.img.setOnLongClickListener(v -> true);

		GlideApp.with(context)
				.load(images.get(position).getImagePath())
				.galleryThumbnail()
				.into(holder.img);
	}

	@Override
	public int getItemCount() {
		return images.size();
	}
}
