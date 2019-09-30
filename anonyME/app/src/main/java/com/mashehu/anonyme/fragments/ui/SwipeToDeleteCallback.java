package com.mashehu.anonyme.fragments.ui;

import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.drawable.ColorDrawable;
import android.graphics.drawable.Drawable;
import android.util.Log;
import android.view.View;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.core.content.ContextCompat;
import androidx.recyclerview.widget.ItemTouchHelper;
import androidx.recyclerview.widget.RecyclerView;

import com.mashehu.anonyme.R;

public class SwipeToDeleteCallback extends ItemTouchHelper.SimpleCallback {
	private ConfirmImageLargeAdapter adapter;

	public SwipeToDeleteCallback(ConfirmImageLargeAdapter adapter) {
		super(ItemTouchHelper.UP, 0);
		this.adapter = adapter;
//		icon = ContextCompat.getDrawable(this.adapter.getContext(),
//				R.drawable.ic_delete_white_36);
//		background = new ColorDrawable(Color.RED);
	}

	@Override
	public void onChildDraw(@NonNull Canvas c, @NonNull RecyclerView recyclerView,
							@NonNull RecyclerView.ViewHolder viewHolder,
							float dX, float dY, int actionState, boolean isCurrentlyActive) {
		final View foregroundView = ((ConfirmImageViewHolder) viewHolder).img;
		getDefaultUIUtil().onDraw(c, recyclerView, foregroundView, dX, dY,
				actionState, isCurrentlyActive);

//		View itemView = viewHolder.itemView;
//		int backgroundCornerOffset = 20;
//		int iconMargin = (itemView.getHeight() - icon.getIntrinsicHeight()) / 2;
//		int iconTop = itemView.getTop() + (itemView.getHeight() - icon.getIntrinsicHeight()) / 2;
//		int iconBottom = iconTop + icon.getIntrinsicHeight();
//
//		if (dY > 0) { // Swiping upwards
//			int iconLeft = itemView.getLeft() + iconMargin + icon.getIntrinsicWidth();
//			int iconRight = itemView.getLeft() + iconMargin;
//			icon.setBounds(iconLeft, iconTop, iconRight, iconBottom);
//			background.setBounds(itemView.getLeft(), itemView.getTop() + (int) dY,
//					itemView.getRight(),
//					itemView.getBottom() + backgroundCornerOffset);
//
//		}
//		else { // view is unSwiped
//			background.setBounds(0, 0, 0, 0);
//		}
//		background.draw(c);
//		icon.draw(c);
	}

	@Override
	public boolean onMove(@NonNull RecyclerView recyclerView, @NonNull RecyclerView.ViewHolder viewHolder, @NonNull RecyclerView.ViewHolder target) {
		return true;
	}

	@Override
	public void onSelectedChanged(@Nullable RecyclerView.ViewHolder viewHolder, int actionState) {
		if (viewHolder != null) {
			final View foregroundView = ((ConfirmImageViewHolder) viewHolder).img;

			getDefaultUIUtil().onSelected(foregroundView);
		}
	}

	@Override
	public void onChildDrawOver(@NonNull Canvas c, @NonNull RecyclerView recyclerView, RecyclerView.ViewHolder viewHolder, float dX, float dY, int actionState, boolean isCurrentlyActive) {
		final View foregroundView = ((ConfirmImageViewHolder) viewHolder).img;
		getDefaultUIUtil().onDrawOver(c, recyclerView, foregroundView, dX, dY,
				actionState, isCurrentlyActive);
	}

	@Override
	public void clearView(@NonNull RecyclerView recyclerView, @NonNull RecyclerView.ViewHolder viewHolder) {
		final View foregroundView = ((ConfirmImageViewHolder) viewHolder).img;
		getDefaultUIUtil().clearView(foregroundView);
	}

	@Override
	public void onSwiped(@NonNull RecyclerView.ViewHolder viewHolder, int direction) {
		int position = viewHolder.getAdapterPosition();
		Log.d("anonyme.SwipeToDeleteCallback", "Swipe");
		adapter.deleteItem(position);
	}
}