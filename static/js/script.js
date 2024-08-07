document.addEventListener('DOMContentLoaded', (event) => {
    // Existing login modal script
    var modal = document.getElementById('loginModal');
    var signinBtn = document.getElementById('signin');
    var span = document.getElementsByClassName('close')[0];

    if (signinBtn) {
        signinBtn.onclick = function() {
            modal.classList.add("block");
            modal.classList.remove("hidden");
            // Show the login form by default when the modal is opened
            document.getElementById('loginForm').style.display = "block";
            document.getElementById('registerForm').style.display = "none";
            document.querySelector('.tab button:first-child').classList.add("active");
        }
    }

    if (span) {
        span.onclick = function() {
            modal.classList.remove("block");
            modal.classList.add("hidden");
        }
    }

    window.onclick = function(event) {
        if (event.target == modal) {
            modal.classList.remove("block");
            modal.classList.add("hidden");
        }
    }

// Default to show the login form
var defaultTab = document.querySelector('.tab button:first-child');
if (defaultTab) {
    defaultTab.click();
}

    // Carousel script
    const track = document.querySelector('.carousel-track');
    const slides = Array.from(track.children);
    const nextButton = document.querySelector('.carousel-button.next');
    const prevButton = document.querySelector('.carousel-button.prev');

    let currentIndex = 0;

    const updateSlidePosition = () => {
        if (track) {
            track.style.transform = 'translateX(-' + (currentIndex * 100) + '%)';
            console.log(`Current Index: ${currentIndex}, Transform: ${track.style.transform}`);
        }
    };

    if (nextButton) {
        nextButton.addEventListener('click', () => {
            if (currentIndex < slides.length - 1) {
                currentIndex++;
            } else {
                currentIndex = 0; // Loop back to the start
            }
            updateSlidePosition();
        });
    }

    if (prevButton) {
        prevButton.addEventListener('click', () => {
            if (currentIndex > 0) {
                currentIndex--;
            } else {
                currentIndex = slides.length - 1; // Loop back to the end
            }
            updateSlidePosition();
        });
    }

    updateSlidePosition(); // Initial call to set position
});

function openForm(evt, formName) {
    var i, tabcontent, tablinks;
    tabcontent = document.getElementsByClassName("tabcontent");
    for (i = 0; i < tabcontent.length; i++) {
        tabcontent[i].style.display = "none";
    }
    tablinks = document.getElementsByClassName("tablinks");
    for (i = 0; i < tablinks.length; i++) {
        tablinks[i].className = tablinks[i].className.replace(" active", "");
    }
    document.getElementById(formName).style.display = "block";
    evt.currentTarget.className += " active";
}
