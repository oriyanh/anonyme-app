package com.mashehu.anonyme.fragments.ui;

import android.view.View;
import android.widget.ImageView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

import com.mashehu.anonyme.R;

public class ConfirmImageViewHolder extends RecyclerView.ViewHolder {
	public ImageView img;

	public ConfirmImageViewHolder(@NonNull View itemView) {
		super(itemView);
		img = itemView.findViewById(R.id.confirm_image_large_view);
	}
}
