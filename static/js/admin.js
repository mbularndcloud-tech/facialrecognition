// static/js/admin.js
// Ensure month and week are exclusive; small helper for admin filters
document.addEventListener('DOMContentLoaded', () => {
  const monthInput = document.getElementById('monthInput');
  const weekInput = document.getElementById('weekInput');

  if (monthInput && weekInput) {
    monthInput.addEventListener('input', () => {
      if (monthInput.value) weekInput.value = '';
    });
    weekInput.addEventListener('input', () => {
      if (weekInput.value) monthInput.value = '';
    });
  }
});

