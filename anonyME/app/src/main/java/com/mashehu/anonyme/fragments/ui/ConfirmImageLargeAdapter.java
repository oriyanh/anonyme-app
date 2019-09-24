package com.mashehu.anonyme.fragments.ui;

import android.content.Context;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;

import androidx.annotation.NonNull;
import androidx.navigation.Navigation;
import androidx.recyclerview.widget.RecyclerView;

import com.mashehu.anonyme.R;

import java.util.ArrayList;

public class ConfirmImageLargeAdapter extends RecyclerView.Adapter<ConfirmImageViewHolder> {
	private ArrayList<ImageData> images;
	private Context context;

	public ConfirmImageLargeAdapter(Context context, ArrayList<ImageData> images) {
		this.images = images;
		this.context = context;
	}



	@NonNull
	@Override
	public ConfirmImageViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
		View view = LayoutInflater.from(parent.getContext()).inflate(R.layout.item_confirm_image_large, parent, false);
		return new ConfirmImageViewHolder(view);	}

	@Override
	public void onBindViewHolder(@NonNull ConfirmImageViewHolder holder, int position) {
//		holder.img.setOnClickListener(v -> {
//			if (!multipleSelectionMode) {
//				Bundle args = new Bundle();
//				String img = images.get(position).getImagePath();
//				ArrayList<String> images = new ArrayList<>();
//				images.add(img);
//				args.putStringArrayList("imageDirs", images);
//				Navigation.findNavController(v).navigate(
//						R.id.action_galleryFragment_to_confirmImagesFragment,
//						args);
//			}
//			else {} //TODO work on multiple selection mode
//		});
//		holder.img.setOnLongClickListener(v -> true);

		GlideApp.with(context)
				.load(images.get(position).getImagePath())
				.confirmImagesLarge()
				.into(holder.img);
	}

	@Override
	public int getItemCount() {
		return images.size();
	}
}
