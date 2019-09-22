package com.mashehu.anonyme.fragments.ui;

import android.view.View;
import android.widget.ImageView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

import com.mashehu.anonyme.R;

public class ThumbnailViewHolder extends RecyclerView.ViewHolder {
	public ImageView img;

	public ThumbnailViewHolder(@NonNull View itemView) {
		super(itemView);
		img = itemView.findViewById(R.id.thumbnail_view);
	}
}
